import math
import traceback
import torch
import torch.distributed as dist
from loguru import logger
from dlx.utils.time import timer
import os

class TrainerMonitor:
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

    def record_rank(self):
        if self.summary_writer is not None and self.enable_record_rank:
            state_dict = self.model.state_dict()
            with torch.no_grad():
                for k in state_dict:
                    # logger.info(f'{k}\t {state_dict[k].shape}')
                    if len(state_dict[k].shape) == 2:
                        torch.cuda.empty_cache()
                        _, S, _ = torch.svd(state_dict[k].cpu(), compute_uv=False)
                        threshold = S.max() * 1e-10
                        rank = torch.sum(S > threshold)
                        self.summary_writer.add_scalar(f'ranks/{k}', rank.item(), global_step=self.cur_step)
                        self.summary_writer.add_histogram(f'eigen_values/{k}', S.cpu().numpy(), global_step=self.cur_step)
                        max_s = S.max().item()
                        min_s = S.min().item()
                        real_rank = torch.sum(S>1e-8).item()
                        self.summary_writer.add_scalar(f'singularity/{k}_max', max_s, global_step=self.cur_step)
                        self.summary_writer.add_scalar(f'singularity/{k}_min', min_s, global_step=self.cur_step)
                        self.summary_writer.add_scalar(f'ranks/{k}_real_rank', real_rank, global_step=self.cur_step)

            torch.cuda.empty_cache()

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
            self.summary_writer.add_scalar(f'{stage}/lr', self.optimizer.param_groups[0]['lr'], self.cur_step)
            if self.tokenizer is not None:
                input_ids = input_x.detach().cpu().numpy()[0]
                input_text = self.tokenizer.decode(input_ids)
                output_ids = output.detach().cpu().numpy()[0].argmax(axis=-1)
                output_text = self.tokenizer.decode(output_ids)
                entire_text = '{:=^40}\n{}\n{:=^40}\n{}'.format('input', input_text, 'output', output_text)
                self.summary_writer.add_text(f'{stage}/inspecting text', entire_text, self.cur_step)
                logger.info('local rank: {}, '.format(os.getenv('LOCAL_RANK', -1)) +
                            'entire text: \n' + entire_text)

    def record_gradient_norm(self, postfix=''):
        if self.summary_writer is not None:  # and (self.cur_step % self.train_log_iters == 0):
            with torch.no_grad():
                for k, v in self.model.named_parameters():
                    if v.grad is not None:
                        grad_norm = torch.norm(v.grad.data.cpu()).item()
                        self.summary_writer.add_scalar(f'grad_norm/{k}{postfix}', grad_norm, self.cur_step)

    def update_params(self):
        _time_begin_optimzer_step = timer.mark()
        # try:
        if next(self.model.parameters()).grad is not None:
            if self.amp:
                if self.grad_clip is not None:
                    try:
                        self.scaler.unscale_(self.optimizer)
                    except Exception as e:
                        logger.error('local rank: {}, {}!'.format(
                            os.getenv('LOCAL_RANK', -1),
                            e
                        ))
                        print(traceback.print_exc())
                    self.record_gradient_norm()
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
                # dist.barrier()
                try:
                    self.scaler.step(self.optimizer)
                except Exception as e:
                    logger.error(f'{e}')
                    for p in self.model.parameters():
                        print(p.grad)
                    import sys
                    sys.exit()
                self.scaler.update()
            else:
                self.record_gradient_norm()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
                # dist.barrier()
                self.optimizer.step()
            _time_end_optimizer = timer.mark()
            self.record_gradient_norm(postfix='_normclip')
            self.optimizer.zero_grad()
            logger.debug('local rank' + str(os.getenv('LOCAL_RANK', -1)) + ', after zero_grad' +
                         f'time of optim: {_time_end_optimizer - _time_begin_optimzer_step}')
        else:
            logger.warning('local rank: {}, step: {}, gradients are None!'.format(
                os.getenv('LOCAL_RANK', -1),
                self.cur_step
            ))
        # except Exception as e:
        #     logger.error('local rank: {}, {}!'.format(
        #         os.getenv('LOCAL_RANK', -1),
        #         e
        #     ))
        #     print(traceback.print_exc())

