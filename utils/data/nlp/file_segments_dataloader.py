import time
import multiprocessing as mp
from dlx.utils.data.dataloader import (
    Dataloader,
    default_generate_batch
)
import glob
import random
from multiprocessing import (
    Value,
    Event,
    Process
)
from loguru import logger
import sys


class FileSegmentsDataloader(Dataloader):
    def __init__(self, dataset_instance,
                 change_file_iters=50000,
                 queue_size=100000,
                 batch_size=4,
                 steps=None,
                 num_worker=8,
                 collate_fn=None,
                 worker_func=None,
                 **kwargs,  # kwargs should only be the arguments for parent class Dataloader
                 ):
        # This is nearly impossible.
        if worker_func is not None:
            self.worker_func = worker_func

        self.files = dataset_instance.files
        self.open_file_func = dataset_instance.open_file_func
        self.change_file_iters = change_file_iters
        self.change_file_event = Event()
        self.change_file_times = Value('i', -1)

        # variables after opening file segment
        self.current_file = None
        self.content_list = list()  # content in file
        self.contents_num = Value('i', 0)

        # variables for changing file
        self.process_count = Value('i', 0)

        # This init method will start the worker watcher which need the self.current_file has been initialized.
        if hasattr(dataset_instance, 'collate_fn'):
            collate_fn = dataset_instance.collate_fn
        super().__init__(
            dataset_instance,
            batch_size=batch_size,
            steps=steps,
            num_worker=num_worker,
            queue_size=queue_size,
            worker_func=self.worker_func,
            collate_fn=collate_fn,
            **kwargs
        )

    def rload_file(self):
        # inform workers exit before changing file, to resolve `list index out of range`
        # if self.workers is None, and make workers_exit_event set,
        # the first batch of workers will be created but not work
        if self.workers is not None \
                and not self.workers_exit_event.is_set():
            self.workers_exit_event.set()

        self.current_file = random.choice(self.files)
        self.content_list = self.open_file_func(self.current_file)

        self.contents_num.value = len(self.content_list)
        # self.change_file_event.set()
        self.change_file_times.value += 1
        logger.debug(f"Loaded file: {self.current_file}, samples: {self.contents_num.value}")

    def arrange_workers(self):
        """触发条件在worker_watcher_func里，当要换文件时才触发"""
        try:
            self.lock.release()
            logger.debug("锁已释放")
        except Exception as e:
            logger.debug("锁无法释放或早已释放")
            logger.debug(str(e))
            pass

        if self.workers is not None:
            if not self.workers_exit_event.is_set():
                self.workers_exit_event.set()
            while True:
                exit_num = 0
                for w, end_event in self.workers:
                    logger.debug(f'{w.name}, w.exit_code: {w.exitcode}, w.is_alive: {w.is_alive()}, end? {end_event.is_set()}')
                    if end_event.is_set() and w.is_alive():
                        logger.debug(f'{w.name}: 触发join')
                        try:
                            w.join(timeout=5)
                        except Exception as e:
                            logger.error(f'{w.name}触发join中发生错误：{e}')
                    if not w.is_alive():
                        exit_num += 1
                if exit_num >= len(self.workers) - 1 :
                    logger.debug("只剩一个进程没有退出")
                    break
                logger.debug(f'waiting for workers exiting...current exit num: {exit_num}')
                logger.debug(f'{self.process_count.value // self.change_file_iters > self.change_file_times.value},'
                             f'{self.process_count.value}, {self.change_file_iters}, {self.change_file_times.value}')
                time.sleep(1)
            self.workers_exit_event.clear()

        # self.workers = [Process(
        #     target=self.worker_func,
        #     # args=(self.workers_exit_event,)
        #     # args=(self.process_count,)
        #     # args=(self.contents_num, self.content_list, self.data_queue, self.process_count, self.lock,
        #     #       self.workers_exit_event, self.tokenizer)
        # ) for i in range(self.num_worker)]
        # for w in self.workers:
        #     w.start()
        self.workers = []
        for i in range(self.num_worker):
            end_event = Event()
            w = Process(target=self.worker_func, args=(end_event,), daemon=True)
            self.workers.append([w, end_event])
            w.start()
        return logger.debug('end of creating processes..................')


    def worker_watcher_func(self):
        try:
            while not self.watcher_exit_event.is_set():

                # do this finally
                # logger.debug(
                #     f'current process_count: {self.dataloader.process_count.value}, data_q_size: {self.dataloader.data_queue.qsize()}')
                if self.current_file is None or \
                        (self.process_count.value // self.change_file_iters > \
                         self.change_file_times.value):
                    self.rload_file()
                    self.arrange_workers()
                    logger.debug(f'end of arranging workers.')
                time.sleep(0.2)
        except Exception as e:
            import traceback
            logger.debug(f'{e}')
            print(traceback.print_exc())
        logger.warning("WorkerWatcher is exiting.")

    def worker_func(self, end_event=None):#contents_num, content_list, queue, process_count, lock, workers_exit, tokenizer):  # 进程
        logger.debug(f'new worker{mp.current_process().name} start.')
        while not self.workers_exit_event.is_set():
            # logger.deubg(f'worker process: {mp.current_process()}, exit_event state: {self.workers_exit_event.is_set()}')
            try:
                if self.contents_num.value == 0:
                    return

                if self.data_queue.qsize()/self.data_queue._maxsize > 0.9:
                    # 防止真的阻塞
                    time.sleep(1)
                    continue
                sample_ind = random.choice(range(self.contents_num.value))
                sample = self.content_list[sample_ind]
                title = sample['title']
                content = sample['content']
                text = '\n'.join([title, content])
                token_ids = self.tokenizer.encode(text, bos=True, eos=True)
                # logger.debug(f'{mp.current_process()}, before putting data_queue, data_queue_size: {self.data_queue.qsize()}/{self.data_queue._maxsize}')
                self.data_queue.put(token_ids)
                # logger.debug(f'{mp.current_process()}, after putting data_queue, and before locking')
                self.lock.acquire()
                # process_count.value+=1
                self.process_count.value += 1
                # logger.debug(f'{mp.current_process()}, after process_count adding')
                self.lock.release()
                # logger.debug(f'{mp.current_process()}, after locking')

                # logger.debug(f"Worker {mp.current_process().name} added data to queue")
            except Exception as e:
                logger.error(str(e))
                pass
        # self.data_queue.cancel_join_thread()  # Kernel code for subprocess exits normally
        logger.debug(f'{mp.current_process().name}, workers_exit_event is set. data queue: {self.data_queue.qsize()}/{self.data_queue._maxsize}')
        end_event.set()
        return
        # mp.current_process().close()

    # def collate_fn(self):
    #     raise NotImplementedError

    #
    # class WorkerWatcher(Thread):
    #     def __init__(self, dataloader_instance, num_proc=8):
    #         super().__init__()
    #         self.dataloader = dataloader_instance
    #
    #     def run(self, *args,):
    #         pass

