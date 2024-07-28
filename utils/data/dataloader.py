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
    Event,
    Pool
)
import random
from loguru import logger
from multiprocessing.managers import BaseManager
import time
from dlx.tokenizer.tiktoken import Tokenizer
import sys

class BaseWatcherThread(Thread):
    def __init__(self):
        super(BaseWatcherThread, self).__init__()
        self.run = self.worker_watcher_func

    def worker_watcher_func(self):
        raise NotImplementedError



# def worker_func(contents_num, content_list, queue, process_count, lock, workers_exit, tokenizer):  # 进程
#     while not workers_exit.is_set():
#         try:
#             if contents_num.value == 0:
#                 return
#             sample_ind = random.choice(range(contents_num.value))
#             sample = content_list[sample_ind]
#             title = sample['title']
#             content = sample['content']
#             text = '\n'.join([title, content])
#             token_ids = tokenizer.encode(text, bos=True, eos=True)
#             queue.put(token_ids)
#             lock.acquire()
#             process_count.value += 1
#             lock.release()
#         except Exception as e:
#             logger.error(str(e))
#             pass

#
# class WorkerWatcher(Thread):
#     def __init__(self, dataloader_instance, num_proc=8):
#         super().__init__()
#         self.dataloader = dataloader_instance
#
#     def run(self, *args,):
#         pass


def default_generate_batch(dataloader_instance, collate_fn=None):
    while not dataloader_instance.watcher_exit_event.is_set():
        if dataloader_instance.debug:
            if not dataloader_instance.data_queue.empty():
                dataloader_instance.data_queue.get()
        else:
            if not dataloader_instance.data_queue.empty():
                batch = []
                for i in range(dataloader_instance.batch_size):
                    sample = dataloader_instance.data_queue.get()
                    batch.append(sample)
                if collate_fn is not None:
                    batch = collate_fn(batch)
                    if (batch is None) or (batch == []):
                        continue
                dataloader_instance.data_list.append(batch)
                dataloader_instance.length += 1



class Dataloader(BaseWatcherThread):
    def __init__(self,
                 dataset,
                 queue_size=2000,
                 batch_size=4,
                 steps=250000,
                 num_worker=8,
                 collate_fn=None,
                 generate_batch_func=default_generate_batch,
                 worker_func=None,
                 worker_watcher=None
                 # num_samples=None
                 ):
        """

        :param dataset:     An instance which is similar with torch.utils.data.Dataset, but the method __len__ can be
            undefined. But the method arrange_workers must be rewritten in this special situation.
        :param queue_size:
        :param batch_size:
        :param steps:
        :param num_worker:
        :param collate_fn:
        :param generate_batch_func:
        :param worker_func:
        :param worker_watcher:
        """
        # init the part of Thread
        super().__init__()

        self.debug = False
        self.dataset_instance = dataset
        self.batch_size = batch_size
        self.steps = steps
        self.num_worker = num_worker
        self.collate_fn = collate_fn
        if worker_func is None:
            worker_func = self.make_dataset_iteration2queue_worker_func(
                self.dataset_instance.__getitem__
            )
        self.worker_func = worker_func

        try:
            self.num_samples = len(dataset)
        except:
            self.num_samples = None

        self.current_step = 0
        # self.process_count = None

        self.manager = Manager()
        self.lock = self.manager.Lock()

        self.data_queue = Queue(maxsize=queue_size)
        self.data_list = list()
        self.length = 0

        self.workers_exit_event = Event()
        self.watcher_exit_event = Event()

        # watch the progress of self.workers to change the content file
        # self.worker_watcher = WorkerWatcher(self, 8)
        if worker_watcher is not None:
            self.worker_watcher = worker_watcher
            self.worker_watcher.start()  # self.process_count.value)
        else:
            # Denoting the worker_watcher as self is considered for development later, not for using now.
            self.worker_watcher = self
            self.start_worker_watcher()

        self.workers = None
        self.tokenizer = Tokenizer()

        # start a thread to take data from queue
        self.generate_batch_thread = Thread(
            target=generate_batch_func,
            args=(self, collate_fn)
        )
        self.generate_batch_thread.start()

    def __getitem__(self, *args):
        while True:
            if self.current_step == self.steps:
                raise StopIteration
            if self.length:
                sample = self.data_list.pop(0)
                self.length -= 1
                self.current_step += 1
                return sample

    def arrange_workers(self):
        self.workers = Pool(self.num_worker)
        for i in range(self.num_samples):
            self.workers.apply_async(
                self.worker_func,
                args=(i,)
            )

    def make_dataset_iteration2queue_worker_func(self, func,):
        def new_func(*args, **kwargs):
            result = func(*args, **kwargs)
            self.data_queue.put(result)
            return result
        return new_func

    def rload_file(self):
        raise NotImplementedError

    def __del__(self):
        logger.warning("正在退出主进程")
        self.workers_exit_event.set()
        self.watcher_exit_event.set()

    def worker_watcher_func(self):
        raise NotImplementedError

    def start_worker_watcher(self):
        try:
            self.start()
        except NotImplementedError:
            pass
        except Exception as e:
            logger.error(e)
            sys.exit(-1)


if __name__ == "__main__":
    root = '/dataset/fd5061f6/chinese_data/WuDao/'
    # dataset = WuDao(root)
    # for i, item in enumerate(dataset):
    #     logger.debug("{}, {}".format(i, item[0]))
