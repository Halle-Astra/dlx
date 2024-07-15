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
    Array
)
import random
from loguru import logger
from multiprocessing.managers import BaseManager

def wf(*args):
    while True:
        worker_func(*args)

def worker_func(contents_num, content_list, queue, process_count):
    try:
        logger.info(f'current_cnus: {contents_num.value}')
        sample_ind = random.choice(range(contents_num.value))
<<<<<<< HEAD
        #logger.info(f'sample_ind : {sample_ind}')
=======
        logger.info(f'sample_ind : {sample_ind}')
>>>>>>> origin/main
        sample = content_list[sample_ind]
        title = sample['title']
        content = sample['content']
        text = '\n'.join([title, content])
        queue.put(text)
        process_count.value += 1
        logger.info('process_count: {}'.format(process_count.value))
    except Exception as e:
        logger.error(str(e))


class WorkerManager(Thread):
    def __init__(self, dataset_instance, num_proc=8):
        super().__init__()
        self.ds = dataset_instance

    def run(self, *args, ):
        logger.debug('inputs are :{}'.format(args))
        while True:
            logger.info('contents is None: {}'.format(self.ds.content_list is None))
<<<<<<< HEAD
            if self.ds.contents_num.value == 0 and (self.ds.process_count.value % self.ds.change_file_iters == 0):
=======
            if self.ds.process_count.value % self.ds.change_file_iters == 0:
>>>>>>> origin/main
                self.ds.rload_file()


class WuDao(BaseManager):
    def __init__(self, root, change_file_iters=2000, queue_size=1000):
        self.files = glob.glob(os.path.join(root, '*.json'))
        self.change_file_iters = change_file_iters
        self.manager = Manager()
        self.current_file = None
        self.content_list = list()
        self.contents_num = Value('i', 0)
        self.data_queue = Queue(maxsize=queue_size)
        self.worker_manager = WorkerManager(self,8)
        self.process_count = Value('i', 0)

        self.worker_manager.start()#self.process_count.value)

<<<<<<< HEAD
        self.workers = [Thread(
=======
        self.workers = [Process(
>>>>>>> origin/main
            target=wf,
            args=(self.contents_num, self.content_list, self.data_queue, self.process_count)
        ) for i in range(8)]
        for w in self.workers:
            w.start()
        self.start()

    def start_worker(self):
        self.rload_file()

    def rload_file(self):
        self.current_file = random.choice(self.files)
        logger.debug(f'current choiced file: {self.current_file}')
        f = open(self.current_file)
<<<<<<< HEAD
        content = json.load(f)
        logger.info('载入完成')
        help(self.content_list)
        self.content_list.clear()
        self.content_list+=content
=======
        cont = json.load(f)
        logger.debug(cont)
        self.content_list = self.manager.list(cont)
>>>>>>> origin/main
        f.close()
        self.contents_num.value = len(self.content_list)

    def worker_func(self):
        pass

    def process_each_item(self, item):
        # dataType, title, content,
        # id 虽然有用，但是id感觉没法一次性全部读入的情况下，还是蛮难用的。
        pass

if __name__ == "__main__":
    root = '/dataset/fd5061f6/chinese_data/WuDao/'
    dataset = WuDao(root)
