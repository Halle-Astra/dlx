import os
import random
import numpy as np
import torch
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
import torch.distributed as dist
from torch import nn
from typing import (
    Callable,
    List
)
from loguru import logger
from dlx.utils.train import save_parameters
from dlx.train.trainer import BaseTrainer


class LossList(nn.Module):
    def __init__(self, loss_list: List):
        super().__init__()
        assert isinstance(loss_list, list), 'Input argument must be a list.'
        self.loss_funcs = nn.ModuleList(list)

    def forward(self, *args, **kwargs):
        loss = 0
        for loss_func in self.loss_funcs:
            loss_value = loss_func(*args, **kwargs)
            if not torch.isnan(loss_value):
                loss = loss + loss_value
        return loss


class DefaultGenerativeLoss(nn.Module):
    def __init__(self):
        super(DefaultGenerativeLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, output, label):
        output = output[:, :-1]
        bs, seq_length, vocab_size = output.shape
        output = output.reshape(-1, vocab_size)
        loss = self.ce_loss(output, label)
        if torch.isnan(loss):
            loss = 0
        return loss


class AutoRegressiveTrainer(BaseTrainer):
    def __init__(self, model, dataloader,
                 loss_module: Callable = DefaultGenerativeLoss(),
                 optimizer=None,
                 world_size=None,
                 tokenizer=None,
                 model_is_kv_cache_enabled=False,
                 device='cuda',
                 dtype=torch.float16,
                 parallel=None,
                 grad_clip=None,
                 start_step=0,
                 save_folder='models_train',
                 epochs=4,
                 train_log_iters=200,
                 eval_log_iters=200,
                 save_iters=20000,
                 eval_dataloader=None):
        """

        :param model:
        :param dataloader:          A dataloader which only generate a batch of list of token ids
        :param loss_modules:
        :param kv_cache_enabled:    determine the training strategy, like GPT if false, or like Llama3 if true, default
                                    value is false.
        """
        # if parallel is not None and parallel == 'ddp':
        #     self.init_parallel()
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        if isinstance(loss_module, list):
            loss_module = LossList(loss_module)

        self.loss_module = loss_module
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.world_size = world_size
        self.grad_clip = grad_clip
        self.model_is_kv_cache_enabled = model_is_kv_cache_enabled
        # self.step = start_step
        self.save_folder = save_folder
        self.epochs = epochs
        self.train_log_iters = train_log_iters
        self.eval_log_iters = eval_log_iters
        self.save_iters = save_iters
        self.eval_dataloader = eval_dataloader

    def init_parallel(self):
        torch.cuda.set_device(dist.get_rank())
        model_parallel_size = self.world_size
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

    def log_training(self, train_loss):
        info_string = [];
        sep = ' | '
        info_string.append(f'step: {self.cur_step}')
        info_string.append(f'loss: {train_loss}')
        info_string = sep.join(info_string)
        logger.info(info_string)

    def start(self):
        for _e in range(self.epochs):
            for i, batch in enumerate(self.dataloader):
                # self.step += 1
                input_x, label, other_args = batch
                input_x = input_x.to(self.device)
                label = label.to(self.device)

                output = self.model(input_x, **other_args)
                loss = self.loss_module(output, label)

                # print(loss.item())

                self.optimizer.zero_grad()
                if loss > 0:
                    loss.backward(retain_graph=True)
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm(
                            self.model.parameters(),
                            self.grad_clip
                        )
                    self.optimizer.step()

                if self.model_is_kv_cache_enabled:
                    self.model.module.reset_kv_cache()

                eval_loss = -1

                # log training states
                if self.cur_step % self.train_log_iters == 0:
                    self.log_training(loss.item())


                # log evaluating states
                if self.cur_step % self.eval_log_iters == 0 and self.eval_dataloader is not None:
                    pass



                # save parameters
                if self.cur_step % self.save_iters == 0 and self.cur_step > 0:
                    self.save(loss, eval_loss)

                self.cur_step += 1


            self.save(loss, eval_loss)
            self.cur_epoch += 1
