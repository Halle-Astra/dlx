from dlx.utils.data.dataloader import (
    Dataloader,
    generate_batch
)
import glob
import random


class FileSegmentsDataloader(Dataloader):
    def __init__(self, dataset_instance,
                 change_file_iters=1000,
                 queue_size=200,
                 batch_size=4,
                 steps=250000,
                 num_worker=8,
                 collate_fn=None,

                 ):
        super().__init__(
            batch_size=batch_size,
            steps=steps,
            num_worker=num_worker,
            queue_size=queue_size,
            collate_fn=collate_fn
        )
        self.files = dataset_instance.files
        self.open_file_func = dataset_instance.open_file_func
        self.change_file_iters = change_file_iters
        self.change_file_times = Value('i', 0)

        # variables after opening file segment
        self.current_file = None
        self.content_list = list()  # content in file
        self.contents_num = Value('i', 0)

        # variables for changing file
        self.process_count = Value('i', 0)
        self.change_file_event = Event()

    def rload_file(self):
        self.current_file = random.choice(self.files)
        self.content_list = self.open_file_func(self.current_file)

        self.contents_num.value = len(self.content_list)
        self.change_file_event.set()
        self.change_file_times.value += 1
        logger.debug(f'Source file {self.current_file} is loaded.')

    def arrange_workers(self):
        try:
            self.lock.release()
            logger.info("锁已释放")
        except Exception as e:
            logger.warning("锁无法释放或早已释放")
            logger.error(str(e))
            pass

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
            target=self.worker_func,
            args=(self.contents_num, self.content_list, self.data_queue, self.process_count, self.lock,
                  self.workers_exit_event, self.tokenizer)
        ) for i in range(self.num_worker)]
        for w in self.workers:
            w.start()
