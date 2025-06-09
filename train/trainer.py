import multiprocessing

from dlx.utils.train import save_parameters
import os
import glob
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
import torch.distributed as dist
import math
import tqdm
from dlx.utils.time import timer
import time
from torch import nn
from dlx.train.llm.monitor import TrainerMonitor
from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy
import traceback


class PlotHelper:
    def __init__(self):
        self.width = 6
        self.height = 4
        self.dpi = 200
        self.fig_convert = plt.figure(figsize=(self.width, self.height), dpi=self.dpi)
        self.axes_convert = self.fig_convert.add_axes([0.16, 0.15, 0.75, 0.75])

    def plot_to_matrix(self, x, y):
        self.axes_convert.cla()
        self.axes_convert.plot(x, y)

        self.fig_convert.canvas.draw()
        fig_str = self.fig_convert.canvas.tostring_argb()
        data = np.frombuffer(fig_str, dtype=np.uint8).reshape((self.height * self.dpi, -1, 4)) / 255.0
        return data

    def __call__(self, x, y):
        return self.plot_to_matrix(x, y)


plot_helper = PlotHelper()


class BaseTrainer(TrainerMonitor):
    def __init__(self, ):
        self.cur_epoch = 0
        self.cur_step = 0
        self.tokens_num = 0
        self.save_folder = None
        self.accumulate_iters = 1

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

    def save(self, train_loss=-1, eval_loss=-1, tokens_num=-1):
        assert self.save_folder is not None, 'save_folder is not set up.'
        folder_name = f'epoch:{self.cur_epoch}-step:{self.cur_step}-train_loss:{train_loss}-eval_loss:{eval_loss}'

        def _save(folder_name):
            folder = os.path.join(self.save_folder, folder_name)
            if isinstance(self.model, DDP):
                model_state_dict = self.model.module.state_dict(),
            else:
                model_state_dict = self.model.state_dict()

            others = dict(cur_step=self.cur_step,
                          cur_epoch=self.cur_epoch,
                          loss=train_loss,
                          eval_loss=eval_loss,
                          tokens_num=tokens_num)
            save_parameters(
                folder,
                model_state_dict,
                self.optimizer.state_dict(),
                others
            )
            logger.info(f'saved weights to {folder}')

        _save(folder_name)
        _save('latest')

    def load_weights(self, weights_path, prefix='', ext='.pth'):
        if os.path.isfile(weights_path):
            weights = [weights_path]
        else:
            file_format = '*'.join([prefix, ext])
            weights = glob.glob(
                os.path.join(weights_path,
                             os.path.sep,
                             file_format)
            )
        weights = [torch.load(i) for i in weights]
        for weight in weights:
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(weight)
            else:
                if isinstance(weight, tuple):
                    weight = weight[0]
                self.model.load_state_dict(weight)

    def resume(self, folder=None, ext='.pth'):
        if folder is None:
            folder = self.save_folder

        if 'latest' in os.listdir(folder):
            folder = os.path.join(folder, 'latest')
            assert os.path.isdir(folder), 'Argument `folder` should be a directory.'

        model_path = os.path.join(folder, 'model' + ext)
        others_path = os.path.join(folder, 'others' + ext)
        optim_path = os.path.join(folder, 'optim' + ext)

        self.load_weights(model_path)

        optim_weights = torch.load(optim_path)
        if isinstance(optim_weights, tuple):
            optim_weights = optim_weights[0]
        self.optimizer.load_state_dict(optim_weights)

        others = torch.load(others_path)
        self.cur_step = others['cur_step']
        self.cur_epoch = others['cur_epoch']
        self.tokens_num = others['tokens_num']

        logger.info(f'loaded weights from {folder}')

    def evaluate(self):
        eval_loss = 0
        ppl_tokens_num = 0
        ppl = 0
        bar = tqdm.tqdm(total=len(self.eval_dataloader))
        torch.cuda.empty_cache()
        for i, batch in enumerate(self.eval_dataloader):
            try:
                input_x, label, o_args = batch
                input_x, label = input_x.to(self.device), label.to(self.device)
                output_temp, loss_temp = self.forward_and_compute_loss(input_x, label, **o_args)
                loss_temp = loss_temp.detach().cpu().item()
                eval_loss += loss_temp / len(self.eval_dataloader)
                ppl_tokens_num_temp = o_args.get('tokens_num', 0) - 1
                ppl_tokens_num += ppl_tokens_num_temp
                ppl += loss_temp * ppl_tokens_num_temp
            except torch.cuda.OutOfMemoryError:
                logger.warning(f'local rank: {os.getenv("LOCAL_RANK", -1)}, OOM in evaluation...')
                torch.cuda.empty_cache()
            finally:
                bar.update(1)

        if dist.is_initialized():
            ppl_tokens_num, ppl = torch.tensor(ppl_tokens_num, device=self.device), torch.tensor(ppl,
                                                                                                 device=self.device)
            dist.all_reduce(ppl_tokens_num);
            dist.all_reduce(ppl)
            ppl_tokens_num, ppl = ppl_tokens_num.cpu().item(), ppl.detach().cpu().item()
            eval_loss = torch.tensor(eval_loss, device=self.device)
            dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
            eval_loss = eval_loss.detach().cpu().item()
            eval_loss /= dist.get_world_size()

        ppl /= ppl_tokens_num
        ppl = math.exp(ppl)
        bar.close()
        self.log(eval_loss, None, input_x, output_temp, None, 'validate', ppl)
        torch.cuda.empty_cache()

    def start(self):
        if self.profile_dir is not None and isinstance(self.profile_dir, str):
            self._start_with_profile()
        else:
            self._start_without_profile()

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

    def skip_processed_data(self, skip_num):
        bar = tqdm.tqdm(total=skip_num)
        bar.set_description('skip processed data')
        for i in range(skip_num):
            next(self.dataloader)
            bar.update(1)
        bar.close()

    def train_step(self, batch, hvars: Dict = {}):
        """
        Behaviours: training with given batch (forward and backward, update params), normalize loss if it is necessary,
                    do logging of training or evaluating, saving params.
        :param batch:
        :param hvars: vars of helper
        :return:
        """
        valid_batch_nums = hvars.get('valid_batch_nums', 0)
        time_mem = hvars.get('time_mem', None)
        _time_wait_batch = hvars.get('time_wait_batch', 0)
        prof = hvars.get('prof', None)
        prof_stopped_flag = hvars.get('prof_stopped_flag', False)
        bar = hvars.get('bar', None)
        loss = None
        try:
            _time_got_batch = timer.mark()
            input_x, label, other_args = batch
            # if dist.is_initialized(): dist.barrier()
            input_x = input_x.to(self.device)
            label = label.to(self.device)

            logger.debug(f'{self.cur_step}, cost of catching batch: {_time_got_batch - _time_wait_batch}s')

            output, loss = self.forward_and_compute_loss(input_x, label, **other_args)
            _time_end_forward = _time_end_loss = timer.mark()
            logger.debug(f'cost of forwarding and computing loss: {_time_end_loss - _time_end_forward}')

            valid_batch_nums += 1 if loss.item() > 0 else 0
            # loss_accumulated = loss + loss_accumulated;
            logger.debug('local rank' + str(os.getenv('LOCAL_RANK', -1)) + ', after loss_accumulated');

            # loss = loss/other_args.get('tokens_num', 1)  # 这会导致loss太小，从而优化步长太小，最后等效于梯度消失
            # normalize loss by max_length
            if self.norm_loss:
                if self.max_length > 0:
                    max_length = self.max_length
                else:
                    max_length = other_args.get('tokens_num', 1)
                loss = loss / other_args.get('tokens_num', 1) * max_length
            if loss > 0: self._backward(loss / self.accumulate_iters)

            if self.cur_step % self.accumulate_iters == 0:  # and loss_accumulated.item() > 0:
                # loss.backward()  # retain_graph=True)
                self.update_params()
                # loss_accumulated = torch.tensor(0, device=self.device)

            _time_end_backward = timer.mark()
            logger.debug(f'cost of backward: {_time_end_backward - _time_end_loss}')

            if self.model_is_kv_cache_enabled: self.model.module.reset_kv_cache()

        except torch.cuda.OutOfMemoryError:
            logger.error(
                f"Local rank: {os.getenv('LOCAL_RANK', -1)} OOM! SAMPLE TOKENS: {other_args.get('tokens_num', -1)}")
        except Exception as e:
            logger.error(
                f"Local rank: {os.getenv('LOCAL_RANK', -1)} ERR with {e}! SAMPLE TOKENS: {other_args.get('tokens_num', -1)}"
            )
            pexc = traceback.print_exc()
            print(pexc)
        finally:
            # other minor operations
            if time_mem is not None: time_mem['batch_cost'].append(_time_got_batch - _time_wait_batch)
            tokens_of_batch = torch.tensor([other_args.get('tokens_num', 0)]).to(self.device)
            if dist.is_initialized(): dist.all_reduce(tokens_of_batch)
            self.tokens_num += tokens_of_batch.item()
            del tokens_of_batch
            self.cur_step += 1

            # log training states
            if self.cur_step % self.train_log_iters == 0:
                self.log(loss, valid_batch_nums, input_x, output, time_mem)
                self.record_rank()
                valid_batch_nums = 0
                if time_mem is not None: time_mem['batch_cost'].clear()

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
            if bar is not None: bar.update(1)
        hvars.update(
            valid_batch_nums=valid_batch_nums,
            time_wait_batch=_time_wait_batch,
            prof_stopped_flag=prof_stopped_flag
        )
        return loss, eval_loss  # , valid_batch_nums, _time_wait_batch, prof_stopped_flag

    def find_lr_and_use_best_lr_to_train(
            self,
            init_value=1e-8,
            final_value=10.,
            beta=0.98,
            num=20,
            first_epoch=False,
            hvars={}
    ):
        """
        Since find_lr must influence the training dataloader and model weights, wrap its main logic and rollback to
        former state to train with subsequent params.
        :param init_value:
        :param final_value:
        :param beta:
        :param num:
        :param first_epoch:
        :param hvars:
        :return:
        """
        # 为了避免find_lr对原模型造成问题，先备份原相关状态，之后确定好了最优lr后重置状态
        fl_begin_step = self.cur_step  # find_lr_begin_step

        def move_state_dict(state_dict, device):
            for key, value in state_dict.items():
                state_dict[key] = value.to(device)

        fl_begin_model_state = deepcopy(self.model.state_dict())
        move_state_dict(fl_begin_model_state, 'cpu')
        torch.cuda.empty_cache()

        training_bar = hvars.get('bar', None)
        hvars['bar'] = None  # 促使find_lr期间，原进度条不受影响

        best_lr, log_lrs, losses, record_batches = self.find_lr(init_value, final_value, beta, num, first_epoch, hvars)
        self.optimizer.param_groups[0]['lr'] = best_lr

        if self.summary_writer is not None:
            self.summary_writer.add_scalar('train/lr', best_lr, self.cur_step)

        # roll back the cur_step
        self.cur_step = fl_begin_step
        hvars['bar'] = training_bar
        self.model.load_state_dict(fl_begin_model_state)
        if hasattr('device', self):
            self.model.to(self.device)
        else:
            self.model.cuda()

        if record_batches:
            for batch in record_batches:
                loss, eval_loss = self.train_step(batch, hvars)
        else:
            loss = None
            eval_loss = -1

        return loss, eval_loss


    def find_lr(self,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                num=20,
                first_epoch=False,
                hvars={}):
        """

        :param init_value:
        :param final_value:
        :param beta:
        :param num:
        :param first_epoch:     if current epoch isn't the first epoch, the max lr will be the 10*current_lr
        :param hvars:           helper variables, Optional
        :return:

        Reference:
            调参-如何确定学习率 lr - 云中江树的文章 - 知乎 https://zhuanlan.zhihu.com/p/559619569
        """
        logger.info(
            f'LOCAL RANK: {os.getenv("LOCAL_RANK", -1)}, FIND LR, init lr: {init_value}, final lr: {final_value}'
        )

        # 确保不会影响到training bar
        restore_training_bar_flag = False
        if 'bar' in hvars and hvars['bar'] is not None:
            training_bar = hvars['bar']
            hvars['bar'] = None
            restore_training_bar_flag = True

        find_lr_bar = tqdm.tqdm(desc='find lr')
        if not first_epoch:
            final_value = 10 * self.optimizer.param_groups[0]['lr']

        # num = len(trn_loader) - 1
        # num = num*self.accumulate_iters

        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []

        bar = tqdm.tqdm(total=num, desc='find lr')
        record_batches = []
        for i in range(num):  # num is the number of optimizer.step, not only forwarding
            loss_tls = []  # temp list
            for j in range(self.accumulate_iters):
                try:
                    batch = next(self.dataloader)
                    record_batches.append(batch)
                except Exception as e:
                    logger.error(f'Generate from dataloader failed: {e}')
                    import traceback
                    print(traceback.print_exc())
                    break

                # As before, get the loss for this mini-batch of inputs/outputs
                loss_i, eval_loss = self.train_step(
                    batch,
                    hvars
                )
                loss_tls.append(loss_i)
                find_lr_bar.update()
            loss = sum(loss_tls) / len([i for i in loss_tls if i > 0])

            batch_num += 1
            # Compute the smoothed loss
            if isinstance(loss, torch.Tensor): loss = loss.item()
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_loss = avg_loss / (1 - beta ** batch_num)

            if self.summary_writer is not None:
                self.summary_writer.add_scalar('find_lr/loss', loss, self.cur_step)
                self.summary_writer.add_scalar('find_lr/avg_loss', avg_loss, self.cur_step)
                self.summary_writer.add_scalar('find_lr/smoothed_loss', smoothed_loss, self.cur_step)
                self.summary_writer.add_scalar('find_lr/lr', lr, self.cur_step)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                break
                # return log_lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))

            # Update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr

            bar.update(1)
        best_lr = 10 ** (log_lrs[np.argmin(losses)])


        if self.summary_writer is not None:
            find_lr_fig = plot_helper(log_lrs, losses)
            self.summary_writer.add_image('find_lr/plot', find_lr_fig, self.cur_step)
        bar.close()

        # 复原traing_bar
        if restore_training_bar_flag or 'training_bar' in locals():
            hvars['bar'] = training_bar
        return best_lr, log_lrs, losses, record_batches
