# my own root: /media/halle/GTA/data/llms/WuDaoCorpus2.0_base_200G
import os
import json
import glob
from threading import Thread
from multiprocessing import (
    Process,
    Queue,
    Manager,
    Value,
    Array,
    Event
)
import random
from loguru import logger
from multiprocessing.managers import BaseManager
import time


def wf(*args):
    while True:
        worker_func(*args)


def worker_func(contents_num, content_list, queue, process_count, lock, workers_exit):  # 进程
    while not workers_exit.is_set():
        try:
            logger.info('contents_num: {}, process_count: {}, data_queue_size:{}'.format(
                contents_num.value,
                process_count.value,
                queue.qsize()
            ))

            if contents_num.value == 0:
                return
            sample_ind = random.choice(range(contents_num.value))
            sample = content_list[sample_ind]
            title = sample['title']
            content = sample['content']
            text = '\n'.join([title, content])
            queue.put(text)
            lock.acquire()
            process_count.value += 1
            lock.release()
        except Exception as e:
            logger.error(str(e))


class WorkerWatcher(Thread):
    def __init__(self, dataset_instance, num_proc=8):
        super().__init__()
        self.ds = dataset_instance

    def run(self, *args, ):
        while True:

            # do this finally
            logger.debug(
                f'current process_count: {self.ds.process_count.value}, data_q_size: {self.ds.data_queue.qsize()}')
            if self.ds.current_file is None or (self.ds.process_count.value % self.ds.change_file_iters == 0):
                self.ds.rload_file()
                self.ds.arrange_workers()
            # time.sleep(10)


class WuDao:
    def __init__(self, root, change_file_iters=1000, queue_size=1000):
        self.files = glob.glob(os.path.join(root, '*.json'))
        self.debug = True

        self.change_file_iters = change_file_iters
        self.manager = Manager()
        self.lock = self.manager.Lock()
        self.current_file = None
        self.content_list = list()
        self.contents_num = Value('i', 0)
        self.data_queue = Queue(maxsize=queue_size)
        self.worker_manager = WorkerWatcher(self, 8)
        self.process_count = Value('i', 0)
        self.change_file_event = Event()
        self.workers_exit_event = Event()

        self.worker_manager.start()  # self.process_count.value)
        # self.rload_file()

        self.workers = None

        self.queue2list_thread = Thread(
            target=self.queue2list,
            args=(self,)
        )
        self.queue2list_thread.start()

    def queue2list(self, dataset_instance):
        while True:
            if dataset_instance.debug:
                # try :
                if dataset_instance.data_queue.qsize() > 0:
                    # logger.info("取数据中, empty:{}, qsize:{}".format(
                    #     dataset_instance.data_queue.empty(),
                    #     dataset_instance.data_queue.qsize()
                    # ))
                    time1 = time.time()
                    dataset_instance.data_queue.get()
                    time2 = time.time()
                    logger.info("time cost of data taking: {}".format(time2 - time1))
                # except Exception as e:
                #     logger.debug(e)

    def arrange_workers(self):
        try:
            self.lock.release()
            logger.info("锁已释放")
        except Exception as e:
            logger.warning("锁无法释放或早已释放")
            logger.error(str(e))

        self.workers_exit_event.set()
        if self.workers is not None:
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
            target=worker_func,
            args=(self.contents_num, self.content_list, self.data_queue, self.process_count, self.lock,
                  self.workers_exit_event)
        ) for i in range(8)]
        for w in self.workers:
            w.start()

    def start_worker(self):
        self.rload_file()

    def rload_file(self):
        self.current_file = random.choice(self.files)
        logger.debug(f'current choiced file: {self.current_file}')
        f = open(self.current_file)
        content = json.load(f)
        logger.info('Source file is changed to {}'.format(self.current_file))
        self.content_list = content
        f.close()
        self.contents_num.value = len(self.content_list)
        self.change_file_event.set()
        logger.debug('Source file is loaded.')


if __name__ == "__main__":
    root = '/dataset/fd5061f6/chinese_data/WuDao/'
    WuDao(root)
