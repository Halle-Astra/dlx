from dlx.utils.data.dataloader import (
    Dataloader,
    generate_batch
)
import glob
import random
from multiprocessing import (
    Value,
    Event,
    Process
)
from loguru import logger


class FileSegmentsDataloader(Dataloader):
    def __init__(self, dataset_instance,
                 change_file_iters=1000,
                 queue_size=200,
                 batch_size=4,
                 steps=250000,
                 num_worker=8,
                 collate_fn=None,
                 worker_func=None
                 ):
        # This is nearly impossible.
        if worker_func is not None:
            self.worker_func = worker_func

        self.files = dataset_instance.files
        self.open_file_func = dataset_instance.open_file_func
        self.change_file_iters = change_file_iters
        self.change_file_times = Value('i', 0)

        # variables after opening file segment
        self.current_file = None
        self.content_list = list()  # content in file
        self.contents_num = Value('i', 0)

        # variables for changing file
        self.process_count = Value('i', 0)
        self.change_file_event = Event()

        # This init method will start the worker watcher which need the self.current_file has been initialized.
        super().__init__(
            dataset_instance,
            batch_size=batch_size,
            steps=steps,
            num_worker=num_worker,
            queue_size=queue_size,
            worker_func=self.worker_func,
            collate_fn=collate_fn
        )

    def rload_file(self):
        self.current_file = random.choice(self.files)
        self.content_list = self.open_file_func(self.current_file)

        self.contents_num.value = len(self.content_list)
        self.change_file_event.set()
        self.change_file_times.value += 1
        logger.debug(f'Source file {self.current_file} is loaded.')

    def arrange_workers(self):
        try:
            self.lock.release()
            logger.info("锁已释放")
        except Exception as e:
            logger.warning("锁无法释放或早已释放")
            logger.error(str(e))
            pass

        if self.workers is not None:
            self.workers_exit_event.set()
            while True:
                exit_num = 0
                for w in self.workers:
                    if not w.is_alive():
                        exit_num += 1
                if exit_num == len(self.workers):
                    logger.info("所有进程已成功退出")
                    break
            self.workers_exit_event.clear()

        self.workers = [Process(
            target=self.worker_func,
            # args=(self.contents_num, self.content_list, self.data_queue, self.process_count, self.lock,
            #       self.workers_exit_event, self.tokenizer)
        ) for i in range(self.num_worker)]
        for w in self.workers:
            w.start()

    def worker_watcher_func(self):
        while not self.watcher_exit_event.is_set():

            # do this finally
            # logger.debug(
            #     f'current process_count: {self.dataloader.process_count.value}, data_q_size: {self.dataloader.data_queue.qsize()}')
            if self.current_file is None or \
                    (
                            self.process_count.value // self.change_file_iters > self.dataloader.change_file_times.value):
                self.rload_file()
                self.arrange_workers()
        logger.warning("WorkerWatcher is exiting.")

    def worker_func(self,):#contents_num, content_list, queue, process_count, lock, workers_exit, tokenizer):  # 进程
        while not self.workers_exit_event.is_set():
            try:
                if self.contents_num.value == 0:
                    return
                sample_ind = random.choice(range(self.contents_num.value))
                sample = self.content_list[sample_ind]
                title = sample['title']
                content = sample['content']
                text = '\n'.join([title, content])
                token_ids = self.tokenizer.encode(text, bos=True, eos=True)
                self.data_queue.put(token_ids)
                self.lock.acquire()
                self.process_count.value += 1
                self.lock.release()
            except Exception as e:
                logger.error(str(e))
                pass

    #
    # class WorkerWatcher(Thread):
    #     def __init__(self, dataloader_instance, num_proc=8):
    #         super().__init__()
    #         self.dataloader = dataloader_instance
    #
    #     def run(self, *args,):
    #         pass

