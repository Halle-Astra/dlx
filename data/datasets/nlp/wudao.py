# my own root: /media/halle/GTA/data/llms/WuDaoCorpus2.0_base_200G
import os
import json
import glob
from threading import Thread
from multiprocessing import (
    Process,
    Queue
)
import random


class WorkerManager(Process):
    def __init__(self, dataset_instance, num_proc=8):
        self.ds = dataset_instance
        self.process_count = 0
        self.processes = [Process(target=self.worker_func, args=(
            self.ds.content_list,
            self.ds.data_queue
        ))]

    def worker_func(self, content_list, queue):
        try:
            sample = random.choice(range(self.ds.contents_num))
            title = sample['title']
            content = sample['content']
            text = '\n'.join([title, content])
            queue.put(text)
            self.process_count += 1
            logger.info('process_count: {}'.format(self.process_count))
        except Exception as e:
            logger.error(str(e))


    def run(self):
        for p in self.processes:
            p.start()
        while True:
            if self.process_count % self.ds.change_file_iters == 0:
                self.ds.rload_file()


class WuDao:
    def __init__(self, root, change_file_iters=2000, queue_size=1000):
        self.files = glob.glob(os.path.join(root, '*.json'))
        self.change_file_iters = change_file_iters
        self.current_file = None
        self.content_list = None
        self.contents_num = None
        self.data_queue = Queue(maxsize=queue_size)
        self.worker_manager = WorkerManager(self,8)
        self.worker_manager.start()

    def start_worker(self):
        self.rload_file()

    def rload_file(self):
        self.current_file = random.choice(self.files)
        f = open(self.current_file)
        self.content_list = json.load(f)
        f.close()
        self.contents_num = len(self.content_list)

    def worker_func(self):
        pass

    def process_each_item(self, item):
        # dataType, title, content,
        # id 虽然有用，但是id感觉没法一次性全部读入的情况下，还是蛮难用的。
        pass
