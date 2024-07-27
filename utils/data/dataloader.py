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


class WorkerWatcher(Thread):
    def __init__(self, dataloader_instance, num_proc=8):
        super().__init__()
        self.dataloader = dataloader_instance

    def run(self, *args,):
        while not self.dataloader.watcher_exit_event.is_set():

            # do this finally
            # logger.debug(
            #     f'current process_count: {self.dataloader.process_count.value}, data_q_size: {self.dataloader.data_queue.qsize()}')
            if self.dataloader.current_file is None or \
                    (
                            self.dataloader.process_count.value // self.dataloader.change_file_iters > self.dataloader.change_file_times.value):
                self.dataloader.rload_file()
                self.dataloader.arrange_workers()
        logger.warning("WorkerWatcher is exiting.")


def generate_batch(dataloader_instance, collate_fn=None):
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



class Dataloader:
    def __init__(self,
                 queue_size=2000,
                 batch_size=4,
                 steps=250000,
                 num_worker=8,
                 collate_fn=None,
                 generate_batch_func=generate_batch,
                 worker_func=None,
                 num_samples=None
                 ):

        self.debug = False

        self.batch_size = batch_size
        self.steps = steps
        self.num_worker = num_worker
        self.collate_fn = collate_fn
        self.worker_func = worker_func
        self.num_samples = num_samples

        self.current_step = 0
        self.process_count = None

        self.manager = Manager()
        self.lock = self.manager.Lock()

        self.data_queue = Queue(maxsize=queue_size)
        self.data_list = list()
        self.length = 0

        self.workers_exit_event = Event()
        self.watcher_exit_event = Event()

        # watch the progress of self.workers to change the content file
        self.worker_watcher = WorkerWatcher(self, 8)
        self.worker_watcher.start()  # self.process_count.value)

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
            Pool.async_apply(self.worker_func, args=(i,))


    def rload_file(self):
        raise NotImplementedError

    def __del__(self):
        logger.warning("正在退出主进程")
        self.workers_exit_event.set()
        self.watcher_exit_event.set()

    def sample_process(self, sample):
        pass


if __name__ == "__main__":
    root = '/dataset/fd5061f6/chinese_data/WuDao/'
    dataset = WuDao(root)
    for i, item in enumerate(dataset):
        logger.debug("{}, {}".format(i, item[0]))
