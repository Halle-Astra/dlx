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
from dlx.tokenizer.tiktoken import Tokenizer


def worker_func(contents_num, content_list, queue, process_count, lock, workers_exit, tokenizer):  # 进程
    while not workers_exit.is_set():
        try:
            # logger.info('contents_num: {}, process_count: {}, data_queue_size:{}'.format(
            #     contents_num.value,
            #     process_count.value,
            #     queue.qsize()
            # ))

            if contents_num.value == 0:
                return
            sample_ind = random.choice(range(contents_num.value))
            sample = content_list[sample_ind]
            title = sample['title']
            content = sample['content']
            text = '\n'.join([title, content])
            token_ids = tokenizer.encode(text, bos=True, eos=True)
            queue.put(token_ids)
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
        while not self.ds.watcher_exit_event.is_set():

            # do this finally
            # logger.debug(
            #     f'current process_count: {self.ds.process_count.value}, data_q_size: {self.ds.data_queue.qsize()}')
            if self.ds.current_file is None or \
                    (self.ds.process_count.value // self.ds.change_file_iters > self.ds.change_file_times.value):
                self.ds.rload_file()
                self.ds.arrange_workers()
        logger.warning("WorkerWatcher is exiting.")


class WuDao:
    def __init__(self, root, change_file_iters=1000, queue_size=2000, batch_size=4, steps=250000):
        self.files = glob.glob(os.path.join(root, '*.json'))
        self.debug = False

        self.change_file_iters = change_file_iters
        self.change_file_times = Value('i', 0)
        self.batch_size = batch_size
        self.steps = steps
        self.current_step = 0

        self.manager = Manager()
        self.lock = self.manager.Lock()
        self.current_file = None
        self.content_list = list()
        self.contents_num = Value('i', 0)
        self.data_queue = Queue(maxsize=queue_size)
        self.data_list = list()
        self.length = 0
        self.process_count = Value('i', 0)
        self.change_file_event = Event()
        self.workers_exit_event = Event()
        self.watcher_exit_event = Event()

        # watch the progress of self.workers to change the content file
        self.worker_watcher = WorkerWatcher(self, 8)
        self.worker_watcher.start()  # self.process_count.value)

        self.workers = None
        self.tokenizer = Tokenizer()

        # start a thread to take data from queue
        self.queue2list_thread = Thread(
            target=self.queue2list,
            args=(self,)
        )
        self.queue2list_thread.start()

    def queue2list(self, dataset_instance):
        while not dataset_instance.watcher_exit_event.is_set():
            if dataset_instance.debug:
                if not dataset_instance.data_queue.empty():
                    dataset_instance.data_queue.get()
            else:
                if not dataset_instance.data_queue.empty():
                    batch = []
                    for i in range(dataset_instance.batch_size):
                        sample = dataset_instance.data_queue.get()
                        batch.append(sample)
                    dataset_instance.data_list.append(batch)
                    dataset_instance.length += 1

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
        try:
            self.lock.release()
            logger.info("锁已释放")
        except Exception as e:
            logger.warning("锁无法释放或早已释放")
            logger.error(str(e))

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
            target=worker_func,
            args=(self.contents_num, self.content_list, self.data_queue, self.process_count, self.lock,
                  self.workers_exit_event, self.tokenizer)
        ) for i in range(8)]
        for w in self.workers:
            w.start()

    def rload_file(self):
        self.current_file = random.choice(self.files)

        f = open(self.current_file)
        self.content_list = json.load(f)
        f.close()

        self.contents_num.value = len(self.content_list)
        self.change_file_event.set()
        self.change_file_times.value += 1
        logger.debug(f'Source file {self.current_file} is loaded.')

    def __del__(self):
        logger.warning("正在退出主进程")
        self.workers_exit_event.set()
        self.watcher_exit_event.set()


if __name__ == "__main__":
    root = '/dataset/fd5061f6/chinese_data/WuDao/'
    dataset = WuDao(root)
    for i, item in enumerate(dataset):
        logger.error("{}, {}".format(i, item[0]))
