import os
import random
import time
import multiprocessing
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch
import math
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from matplotlib import pyplot as plt
import traceback
import torch.distributed as dist
from torch import nn
from typing import (
    Callable,
    List
)
from loguru import logger
from dlx.utils.train import save_parameters
from dlx.train.trainer import BaseTrainer
from dlx.utils.time import timer
from torch.nn.parallel import DistributedDataParallel as DDP
import tqdm




class LossList(nn.Module):
    def __init__(self, loss_list: List):
        super().__init__()
        assert isinstance(loss_list, list), 'Input argument must be a list.'
        self.loss_funcs = nn.ModuleList(list)

    def forward(self, *args, **kwargs):
        loss = torch.tensor(0)
        for loss_func in self.loss_funcs:
            loss_value = loss_func(*args, **kwargs)
            if not torch.isnan(loss_value):
                loss = loss.to(loss_value.device) + loss_value
        return loss


class DefaultGenerativeLoss(nn.Module):
    def __init__(self):
        super(DefaultGenerativeLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, output, label):
        output = output[:, :-1]
        label = label.flatten()
        bs, seq_length, vocab_size = output.shape
        output = output.contiguous().view(-1, vocab_size)
        logger.debug('shape, dtype of output: {}, {}'.format(output.shape, output.dtype))
        logger.debug('shape, dtype of label: {}, {}'.format(label.shape, label.dtype))
        loss = self.ce_loss(output, label)
        if torch.isnan(loss):
            loss = torch.tensor(0, device=output.device)
        return loss


class AutoRegressiveTrainer(BaseTrainer):
    def __init__(self, model, dataloader,
                 loss_module: Callable = DefaultGenerativeLoss(),
                 optimizer=None,
                 world_size=None,
                 tokenizer=None,
                 model_is_kv_cache_enabled=False,
                 device='cuda',
                 ids_dtype=torch.float16,
                 parallel=None,
                 grad_clip: float = None,
                 start_step=0,
                 save_folder='models_train',
                 epochs=4,
                 train_log_iters=2000,
                 eval_log_iters=200000,
                 accumulate_iters=1,
                 save_iters=200000,
                 eval_dataloader=None,
                 amp=False,
                 model_parallel_size=None,
                 profile_dir=None,
                 profile_steps=None,
                 vocab_size=None,
                 summary_writer=None,
                 resume=False,
                 enable_record_rank=False,
                 schedule_lr_iters=-1,
                 norm_loss=False,
                 max_length=-1,
                 enable_find_lr=False,
                 record_bad_batch_folder=''
                 ):
        """

        :param model:
        :param dataloader:          A dataloader which only generate a batch of list of token ids
        :param loss_modules:
        :param kv_cache_enabled:    determine the training strategy, like GPT if false, or like Llama3 if true, default
                                    value is false.
        :param enable_find_lr:      When it is enabled, a best lr will be detected and replace the lr of optimizer.
        :param record_bad_batch_folder
                                    If it is set as blank string, the logic of recording bad batches will be
                                    disabled. Default is disabled.
        """
        # if parallel is not None and parallel == 'ddp':
        #     self.init_parallel()
        super().__init__()
        self.model = model.cuda()
        self.dataloader = dataloader

        if isinstance(loss_module, list):
            loss_module = LossList(loss_module)

        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = ids_dtype
        self.world_size = world_size
        self.grad_clip = grad_clip
        self.loss_module = loss_module.to(self.device)
        self.model_is_kv_cache_enabled = model_is_kv_cache_enabled
        # self.step = start_step
        self.save_folder = save_folder
        self.epochs = epochs
        self.train_log_iters = train_log_iters
        self.eval_log_iters = eval_log_iters
        self.save_iters = save_iters
        self.accumulate_iters = accumulate_iters
        self.eval_dataloader = eval_dataloader
        self.amp = amp
        self.profile_dir = profile_dir
        self.profile_steps = profile_steps
        self.vocab_size = vocab_size
        self.schedule_lr_iters = schedule_lr_iters
        self.norm_loss = norm_loss
        self.max_length = max_length
        self.enable_find_lr = enable_find_lr

        self.summary_writer = summary_writer
        self.enable_record_rank = enable_record_rank
        self.record_bad_batch_folder = record_bad_batch_folder
        self.batches_for_recording_bad_samples = [] 

        if amp:
            self.scaler = GradScaler()
        if resume:
            self.resume()

        self.init_parallel(model_parallel_size)
        self.record_rank()

    def forward_and_compute_loss(self, x, label, *args, **other_input_args):
        input_x = x
        other_args = other_input_args
        if not self.amp:
            output = self.model(input_x, **other_args)
            loss = self.loss_module(output, label)
        else:
            with autocast():
                output = self.model(input_x, **other_args)
                loss = self.loss_module(output, label)
        return output, loss

    def _start_main_routine(self, prof=None):
        prof_stopped_flag = False
        valid_batch_nums = 0
        _time_mem = {'batch_cost': []}
        # loss_accumulated = torch.tensor(0, device=self.device)
        skip_num = self.cur_step % len(self.dataloader)
        self.original_dataloader = self.dataloader
        # self.dataloader = iter(self.dataloader)
        # self.skip_processed_data(skip_num)
        skipped_flag = False
        for _e in range(self.epochs):
            _time_wait_batch = timer.mark()
            # bar = tqdm.tqdm(total=self.dataloader.steps)
            bar = tqdm.tqdm(total=len(self.dataloader))
            bar.set_description('training')
            bar.update(skip_num)
            helper_vars = dict(
                valid_batch_nums=valid_batch_nums,
                time_mem=_time_mem,
                time_wait_batch=_time_wait_batch,
                prof=prof,
                prof_stopped_flag=prof_stopped_flag,
                bar=bar
            )

            self.dataloader = iter(self.original_dataloader)  # 每次执行一次iter就会重置一次内部的起始index
            if not skipped_flag:
                self.skip_processed_data(skip_num)
                skipped_flag = True
            if self.enable_find_lr and self.cur_epoch == 0 and skip_num == 0:
                self.find_lr_and_use_best_lr_to_train(first_epoch=True, hvars=helper_vars)
            # for i, batch in enumerate(self.dataloader):
            for i in range(len(self.dataloader)):
                try:
                    batch = next(self.dataloader)
                # except StopIteration:
                #     break
                except Exception as e:
                    logger.warning(f'Meet the error: {e} when catching batch... Plz check the error')
                    print(traceback.print_exc())
                    break

                loss, eval_loss = self.train_step(
                    batch,
                    helper_vars
                )
                if self.schedule_lr_iters > 0 and self.cur_step % self.schedule_lr_iters == 0:
                    # transfer function find_lr to schedule lr
                    loss, eval_loss = self.find_lr_and_use_best_lr_to_train(hvars=helper_vars)

            self.save(loss, eval_loss, tokens_num=self.tokens_num)
            self.cur_epoch += 1
            bar.close()
        prof_stopped_flag = helper_vars.get('prof_stopped_flag', prof_stopped_flag)
        prof.stop() if prof is not None and not prof_stopped_flag else ...
