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

def worker_func(contents_num, content_list, queue, process_count, lock):  # 进程
    try:
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
        logger.info('contents_num: {}, process_count: {}'.format(contents_num.value,
                                                                 process_count.value))
        lock.release()
    except Exception as e:
        logger.error(str(e))


class WorkerManager(Thread):
    def __init__(self, dataset_instance, num_proc=8):
        super().__init__()
        self.ds = dataset_instance

    def run(self, *args, ):
        while True:
            if self.ds.debug:
                try :
                    self.ds.data_queue.get(timeout=5)
                except Exception as e:
                    logger.debug(e)
            logger.debug(f'current process_count: {self.ds.process_count.value}, data_q_size: {self.ds.data_queue.qsize()}')
            if self.ds.current_file is None or \
                    (self.ds.process_count.value > self.ds.change_file_iters):
                self.ds.rload_file()
                self.ds.arrange_workers()
            time.sleep(10)


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
        self.worker_manager = WorkerManager(self,8)
        self.process_count = Value('i', 0)
        self.change_file_event = Event()

        self.worker_manager.start()#self.process_count.value)
        # self.rload_file()

        self.workers = None

    def arrange_workers(self):
        if self.workers is not None:
            for w in self.workers:
                w.join()
                w.terminate()
                w.close()
        self.workers = [Process(
            target=wf,
            args=(self.contents_num, self.content_list, self.data_queue, self.process_count, self.lock)
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

    def worker_func(self):
        pass

    def process_each_item(self, item):
        # dataType, title, content,
        # id 虽然有用，但是id感觉没法一次性全部读入的情况下，还是蛮难用的。
        pass

if __name__ == "__main__":
    root = '/dataset/fd5061f6/chinese_data/WuDao/'
    WuDao(root)
