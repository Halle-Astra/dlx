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


class Dataset:
    def __init__(self, num_workers=8):
        self.nw = num_workers

    def init_processes(self):
        self.processes = []


class WorkerManager(Process):
    def __init__(self, dataset_instance):
        self.ds = dataset_instance
        self.process_count = 0

    def worker_func(self, content_list, queue):
        sample = random.choice(range(self.ds.contents_num))

    def run(self):
        while True:
            if self.process_count % self.ds.change_file_iters == 0:
                self.ds.rload_file()


class WuDao:
    def __init__(self, root, change_file_iters=2000):
        self.files = glob.glob(os.path.join(root, '*.json'))
        self.change_file_iters = change_file_iters
        self.current_file = None
        self.content_list = None
        self.contents_num = None

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
