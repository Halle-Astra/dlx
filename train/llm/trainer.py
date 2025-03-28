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
                 resume=False
                 ):
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

        self.summary_writer = summary_writer

        if amp:
            self.scaler = GradScaler()
        if resume:
            self.resume()

        self.init_parallel(model_parallel_size)

    def init_parallel(self, model_parallel_size=None):
        # if self.world_size > 1:
        #     torch.cuda.set_device(dist.get_rank())
        #     model_parallel_size = self.world_size if model_parallel_size is None else model_parallel_size
        #     if not torch.distributed.is_initialized():
        #         torch.distributed.init_process_group("nccl")
        #     if not model_parallel_is_initialized():
        #         if model_parallel_size is None:
        #             model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        #         initialize_model_parallel(model_parallel_size)

        # if self.world_size > 1:
        if dist.is_initialized(): self.model = DDP(self.model.to(self.device))
        local_rank = dist.get_rank() if dist.is_initialized() else -1
        self.local_rank = local_rank

    def log_with_logger(self, train_loss, valid_batch_ratio=None, batch_cost=None,):
        info_string = [];
        sep = ' | '
        # info_string.append(f'step: {self.cur_step}/{self.dataloader.steps}')
        if dist.is_initialized(): info_string.append(f'local rank: {dist.get_rank()}')
        info_string.append(f'step: {self.cur_step}/{len(self.dataloader)}')
        info_string.append(f'loss: {train_loss}')
        info_string.append(
            f'ratio of valid batches: {valid_batch_ratio * 100}%'
        ) if valid_batch_ratio is not None else ...
        info_string.append(
            'max waiting batch: {:.3f}s'.format(
                max(batch_cost)
                # sum(batch_cost)/len(batch_cost))
            )
        ) if batch_cost is not None and batch_cost else ...
        info_string = sep.join(info_string)
        logger.info(info_string)

    def update_params(self):
        _time_begin_optimzer_step = timer.mark()
        if self.amp:
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
            # dist.barrier()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
            # dist.barrier()
            self.optimizer.step()
        _time_end_optimizer = timer.mark()
        self.optimizer.zero_grad()
        logger.debug('local rank' + str(os.getenv('LOCAL_RANK', -1)) + ', after zero_grad' +
                     f'time of optim: {_time_end_optimizer - _time_begin_optimzer_step}')

    def _backward(self, loss):
        if self.amp:
            _time_begin_compute_grad = timer.mark()
            self.scaler.scale(loss).backward()
            _time_end_compute_grad = timer.mark()
            logger.debug(f'time of grad cal: {_time_end_compute_grad - _time_begin_compute_grad}')

        else:
            _time_begin_compute_grad = timer.mark()
            loss.backward()
            _time_end_compute_grad = timer.mark()
            logger.debug(f'time of grad cal: {_time_end_compute_grad - _time_begin_compute_grad}')

    def _start_without_profile(self):
        self._start_main_routine()
        # self._start_debug()

    def _start_with_profile(self, ):
        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profile_dir),
                record_shapes=True, profile_memory=False,
                with_stack=True) as prof:
            self._start_main_routine(prof)

    def start(self):
        if self.profile_dir is not None and isinstance(self.profile_dir, str):
            self._start_with_profile()
        else:
            self._start_without_profile()

    def log(self, loss, valid_batch_nums, input_x, output, time_mem=None, stage='train', ppl=None):
        if isinstance(loss, torch.Tensor):
            loss_show = None if loss.item() == 0 else loss.item()
        else:
            loss_show = loss
        if stage == 'train':
            valid_batch_ratio = valid_batch_nums / self.train_log_iters if self.cur_step > 0 else None
        else:
            valid_batch_ratio = -1
        if time_mem is None: time_mem = dict(batch_cost=None)
        self.log_with_logger(
            loss_show,
            valid_batch_ratio,
            time_mem['batch_cost']
        )

        if self.summary_writer is not None and loss_show is not None and valid_batch_ratio is not None:
            if ppl is None: ppl = math.exp(loss.detach().cpu().item())
            self.summary_writer.add_scalar(f'{stage}/loss', loss_show, self.cur_step)
            self.summary_writer.add_scalar(f'{stage}/ppl', ppl, self.cur_step)
            self.summary_writer.add_scalar(f'{stage}/valid batch ratio', valid_batch_ratio, self.cur_step)
            self.summary_writer.add_scalar(f'{stage}/tokens num', self.tokens_num, self.cur_step)
            if self.tokenizer is not None:
                input_ids = input_x.detach().cpu().numpy()[0]
                input_text = self.tokenizer.decode(input_ids)
                output_ids = output.detach().cpu().numpy()[0].argmax(axis=-1)
                output_text = self.tokenizer.decode(output_ids)
                entire_text = '{:=^40}\n{}\n{:=^40}\n{}'.format('input', input_text, 'output', output_text)
                self.summary_writer.add_text(f'{stage}/inspecting text', entire_text, self.cur_step)
                logger.info('local rank: {}, '.format(os.getenv('LOCAL_RANK', -1)) +
                            'entire text: \n' + entire_text)

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

    def evaluate(self):
        eval_loss = 0
        ppl_tokens_num = 0
        ppl = 0
        bar = tqdm.tqdm(total=len(self.eval_dataloader))
        for i, batch in enumerate(self.eval_dataloader):
            input_x, label, o_args = batch
            input_x, label = input_x.to(self.device), label.to(self.device)
            output_temp, loss_temp = self.forward_and_compute_loss(input_x, label, **o_args)
            loss_temp = loss_temp.detach().cpu().item()
            eval_loss += loss_temp/ len(self.eval_dataloader)
            ppl_tokens_num_temp = o_args.get('tokens_num', 0) - 1
            ppl_tokens_num += ppl_tokens_num_temp
            ppl += loss_temp * ppl_tokens_num_temp
            bar.update(1)
        if dist.is_initialized():
            ppl_tokens_num, ppl = torch.tensor(ppl_tokens_num, device=self.device), torch.tensor(ppl, device=self.device)
            dist.all_reduce(ppl_tokens_num); dist.all_reduce(ppl)
            ppl_tokens_num, ppl = ppl_tokens_num.cpu().item(), ppl.detach().cpu().item()
            eval_loss = torch.tensor(eval_loss, device=self.device)
            dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
            eval_loss = eval_loss.detach().cpu().item()
            eval_loss /= dist.get_world_size()

        ppl /= ppl_tokens_num
        ppl = math.exp(ppl)
        bar.close()
        self.log(eval_loss, None, input_x, output_temp, None, 'validate', ppl)

    def _start_main_routine(self, prof=None):
        prof_stopped_flag = False
        valid_batch_nums = 0
        _time_mem = {'batch_cost': []}
        loss_accumulated = torch.tensor(0, device=self.device)

        for _e in range(self.epochs):
            _time_wait_batch = time.time()
            # bar = tqdm.tqdm(total=self.dataloader.steps)
            bar = tqdm.tqdm(total=len(self.dataloader))
            skip_num = self.cur_step % len(self.dataloader)
            bar.update(skip_num)
            for i, batch in enumerate(self.dataloader):
                try:
                    if i < skip_num:
                        logger.debug('local rank: {}, skip logic, skip_num: {}, current i: {}'.format(
                            os.getenv('LOCAL_RANK', -1),
                            skip_num,
                            i
                        ))
                        continue
                    if dist.is_initialized(): dist.barrier()
                    # self.step += 1
                    input_x, label, other_args = batch
                    input_x = input_x.to(self.device)
                    label = label.to(self.device)

                    _time_got_batch = timer.mark()
                    logger.debug(
                        f'{self.cur_step}, cost of catching batch: {_time_got_batch - _time_wait_batch}s')  # logger.debug(f'{self.cur_step}, the rest of batches: {self.dataloader._data_list_length}')                # logger.debug(f'the shape of input_x is {input_x.shape}')

                    output, loss = self.forward_and_compute_loss(input_x, label, **other_args)
                    _time_end_forward = _time_end_loss = timer.mark()
                    logger.debug(f'cost of forwarding and computing loss: {_time_end_loss - _time_end_forward}')
                    # print(loss.item())

                    valid_batch_nums += 1 if loss.item() > 0 else 0
                    # loss_accumulated = loss + loss_accumulated;
                    logger.debug('local rank' + str(os.getenv('LOCAL_RANK', -1)) + ', after loss_accumulated');

                    if loss > 0: self._backward(loss / self.accumulate_iters)

                    if self.cur_step % self.accumulate_iters == 0:  # and loss_accumulated.item() > 0:
                        # loss.backward()  # retain_graph=True)
                        self.update_params()
                        # loss_accumulated = torch.tensor(0, device=self.device)

                    _time_end_backward = timer.mark()
                    logger.debug(f'cost of backward: {_time_end_backward - _time_end_loss}')

                    if self.model_is_kv_cache_enabled:
                        self.model.module.reset_kv_cache()

                except torch.cuda.OutOfMemoryError:
                    logger.error(f"Local rank: {os.getenv('LOCAL_RANK', -1)} OOM! SAMPLE TOKENS: {other_args.get('tokens_num', -1)}")
                finally:
                    # other minor operations
                    _time_mem['batch_cost'].append(_time_got_batch - _time_wait_batch)
                    tokens_of_batch = torch.tensor([other_args.get('tokens_num', 0)]).to(self.device)
                    if dist.is_initialized(): dist.all_reduce(tokens_of_batch)
                    self.tokens_num += tokens_of_batch.item()
                    del tokens_of_batch
                    self.cur_step += 1

                    # log training states
                    if self.cur_step % self.train_log_iters == 0:
                        self.log(loss, valid_batch_nums, input_x, output, _time_mem)
                        valid_batch_nums = 0
                        _time_mem['batch_cost'].clear()

                    # log evaluating states
                    eval_loss = -1
                    if self.cur_step % self.eval_log_iters == 0 and self.eval_dataloader is not None:
                        self.evaluate()

                    # profiling
                    prof.step() if prof is not None else ...
                    if prof is not None and self.cur_step > self.profile_steps:
                        prof.stop()
                        prof_stopped_flag = True

                    _time_wait_batch = timer.mark()
                    logger.debug(f'total cost time: {_time_wait_batch - _time_got_batch}')

                    # save parameters
                    if self.cur_step % self.save_iters == 0 and self.cur_step > 0:
                        self.save(loss, eval_loss, tokens_num=self.tokens_num)

                    # checking processes
                    active_processes = multiprocessing.active_children()
                    logger.debug(f"Active processes: {len(active_processes)}")
                    bar.update(1)
            self.save(loss, eval_loss, tokens_num=self.tokens_num)
            self.cur_epoch += 1
            bar.close()
        prof.stop() if prof is not None and not prof_stopped_flag else ...

    def _start_debug(self, prof=None):
        prof_stopped_flag = False
        valid_batch_nums = 0
        _time_mem = {'batch_cost': []}
        loss_accumulated = torch.tensor(0, device=self.device)
        ce_loss = nn.CrossEntropyLoss()

        for _e in range(self.epochs):
            _time_wait_batch = time.time()
            for i, batch in enumerate(self.dataloader):
                # self.step += 1
                pass
                # input_x, label, other_args = batch
                # input_x = input_x.to(self.device)
                # label = label.to(self.device)
                #
                # output = self.model(input_x, **other_args)
                # # bs,seq,vocab_size = output.shape
                # output = output[:, :-1].reshape(-1, self.vocab_size)
                # loss = ce_loss(output, label)
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()
