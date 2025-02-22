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
import math
import traceback


DEFAULT_TRAIN_STEPS = 250000


class BaseWatcherThread(Thread):
    def __init__(self):
        # self.daemon=True
        super(BaseWatcherThread, self).__init__(daemon=True)
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
    """
    fetch elements from dataloader_instance.data_queue and put them into dataloader_instance.data_list (batch list)
    :param dataloader_instance:
    :param collate_fn:
    :return:
    """
    _take_data_minimal_num = 4
    _take_data_min_num_multiplier = 4
    _sleep_time = 3
    _first_flag = True
    _time_first_begin = time.time()
    try:
        while not dataloader_instance.watcher_exit_event.is_set():
            if dataloader_instance.debug:
                if not dataloader_instance.data_queue.empty():
                    dataloader_instance.data_queue.get()
            else:
                v = (not dataloader_instance._data_list_length > _take_data_min_num_multiplier * _take_data_minimal_num and
                            # dataloader_instance.data_queue.qsize() >= dataloader_instance.batch_size
                        not dataloader_instance.data_queue.empty())
                logger.debug(f'{dataloader_instance._data_list_length}, '
                             f'{_take_data_min_num_multiplier * _take_data_minimal_num},'
                             f'{dataloader_instance.data_queue.qsize()},'
                             f'{v}')
                if _first_flag or (
                        not dataloader_instance._data_list_length >
                            _take_data_min_num_multiplier * _take_data_minimal_num
                        and
                            # dataloader_instance.data_queue.qsize() >= dataloader_instance.batch_size
                        not dataloader_instance.data_queue.empty()  # kernel code
                ):
                    batch = []
                    logger.debug('current_data_que size: {}'.format(
                        dataloader_instance.data_queue.qsize()
                    ))
                    for i in range(dataloader_instance.batch_size):
                    # for i in range(min(dataloader_instance.batch_size, dataloader_instance.data_queue.qsize())):
                        sample = dataloader_instance.data_queue.get()
                        batch.append(sample)
                    if collate_fn is not None:
                        batch = collate_fn(batch)
                        if (batch is None) or (batch == []):
                            continue
                    dataloader_instance.data_list.append(batch)
                    dataloader_instance._data_list_length += 1

                    logger.debug(f"Generated batch, data_list length: {dataloader_instance._data_list_length}")

                else:  # Can sleep few time since the data_queue is empty, that's no matter to sleep.
                    _length_begin_sleep = dataloader_instance._data_list_length
                    time.sleep(_sleep_time)
                    _length_end_sleep = dataloader_instance._data_list_length
                    _take_data_minimal_num = _length_begin_sleep - _length_end_sleep
                    logger.debug(f'take_data_min_num: {_take_data_minimal_num}')

                if _first_flag and (dataloader_instance._data_list_length > _take_data_minimal_num):
                    _first_flag = False
        logger.debug('居然触发了退出机制？')
    except Exception as e:
        logger.error(f'{e}')
        traceback_str = traceback.format_exc()
        print(traceback_str)


class Dataloader(BaseWatcherThread):
    def __init__(self,
                 dataset,
                 queue_size=100000,
                 batch_size=4,
                 steps=None,
                 num_worker=8,
                 collate_fn=None,
                 generate_batch_func=default_generate_batch,
                 worker_func=None,
                 worker_watcher=None,
                 # num_samples=None,
                 # as_train_as_test=False,
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
        # self.as_train_as_test = as_train_as_test
        self.dataset_instance = dataset
        self.batch_size = batch_size
        if steps is not None:
            self.steps = steps
        else:
            try:
                sample_num = len(dataset)
                steps = math.floor(sample_num / batch_size)
                self.steps = int(steps)
            except:
                self.steps = DEFAULT_TRAIN_STEPS
                logger.warning('__len__ is not implemented in dataset instance, training steps is set to defalut value.')
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
        self.data_list = list()  # batch list
        self._data_list_length = 0

        self.workers_exit_event = Event()
        self.watcher_exit_event = Event()

        self.workers = None


        # watch the progress of self.workers to change the content file
        # self.worker_watcher = WorkerWatcher(self, 8)
        if worker_watcher is not None:
            self.worker_watcher = worker_watcher
            self.worker_watcher.start()  # self.process_count.value)
        else:
            # Denoting the worker_watcher as self is considered for development later, not for using now.
            self.worker_watcher = self
            self.start_worker_watcher()

        self.tokenizer = Tokenizer()

        # start a thread to take data from queue
        self.generate_batch_thread = Thread(
            target=generate_batch_func,
            args=(self, collate_fn),

        )
        self.generate_batch_thread.start()

    def __getitem__(self, *args):
        _time_begin_getitem = time.time()
        while True:
            logger.debug('try to return batch to trainer')
            if self.current_step == self.steps:
                raise StopIteration
            if self._data_list_length:
                sample = self.data_list.pop(0)
                self._data_list_length -= 1
                self.current_step += 1
                _time_end_getitem = time.time()
                logger.debug(f'the time for waiting batch: {_time_end_getitem - _time_begin_getitem}s')
                return sample
            else:
                logger.debug('empty batch list')
                time.sleep(0.1)

    def arrange_workers(self):
        raise NotImplementedError
        # self.workers = Pool(self.num_worker)
        # for i in range(self.num_samples):
        #     self.workers.apply_async(self.worker_func, args=(i,))

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
            logger.error(f'{e} | start worker watcher failed.')
            sys.exit(-1)


if __name__ == "__main__":
    root = '/dataset/fd5061f6/chinese_data/WuDao/'
    # dataset = WuDao(root)
    # for i, item in enumerate(dataset):
    #     logger.debug("{}, {}".format(i, item[0]))
