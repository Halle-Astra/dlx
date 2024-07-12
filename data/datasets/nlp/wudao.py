# my own root: /media/halle/GTA/data/llms/WuDaoCorpus2.0_base_200G
import os
import json
import glob
from threading import Thread
from multiprocessing import Process


class Dataset:
    def __init__(self, num_workers=8):
        self.nw = num_workers

    def init_processes(self):
        self.processes = []


class WuDao:
    def __init__(self, root, change_file_iters=2000):
        self.files = glob.glob(os.path.join(root, '*.json'))
        self.current_file = None
        self.content_list = None


    def process_each_item(self, item):
        # dataType, title, content,
        # id 虽然有用，但是id感觉没法一次性全部读入的情况下，还是蛮难用的。
        pass

