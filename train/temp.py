import multiprocessing
import json
from pathlib import Path
import pickle
import shutil

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
from typing import Dict, List, Optional, Set, Any
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

        # 新增：异常样本记录相关属性
        self.skip_anomaly_samples = False  # 是否启用跳过异常样本的flag
        self.anomaly_threshold = 2.0  # 异常检测阈值（loss变化倍数）
        self.anomaly_samples_file = "anomaly_samples.json"  # 异常样本记录文件
        self.anomaly_samples: Set[str] = set()  # 异常样本哈希集合
        self.previous_losses: List[float] = []  # 记录最近几个loss值用于异常检测
        self.loss_window_size = 10  # 用于异常检测的loss窗口大小

        # 新增：异常样本存储目录
        self.anomaly_data_folder = "anomaly_samples_data"
        self.anomaly_info_folder = "anomaly_samples_info"
        self.new_anomaly_samples: Set[str] = set()  # 本次运行新发现的异常样本

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

        # 保存异常样本记录（增量保存）
        self._save_anomaly_samples(incremental=True)

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

        # 新增：加载异常样本记录
        self._load_anomaly_samples()

    def _get_anomaly_base_path(self) -> str:
        """获取异常样本存储的基础路径"""
        if self.save_folder is None:
            return None
        return os.path.join(self.save_folder, self.anomaly_data_folder)

    def _get_anomaly_info_path(self) -> str:
        """获取异常样本信息存储路径"""
        if self.save_folder is None:
            return None
        return os.path.join(self.save_folder, self.anomaly_info_folder)

    def _ensure_anomaly_dirs(self):
        """确保异常样本存储目录存在"""
        base_path = self._get_anomaly_base_path()
        info_path = self._get_anomaly_info_path()

        if base_path:
            os.makedirs(base_path, exist_ok=True)
        if info_path:
            os.makedirs(info_path, exist_ok=True)

    def _load_anomaly_samples(self):
        """加载异常样本记录"""
        info_path = self._get_anomaly_info_path()
        if not info_path:
            return

        anomaly_file = os.path.join(info_path, self.anomaly_samples_file)
        if os.path.exists(anomaly_file):
            try:
                with open(anomaly_file, 'r') as f:
                    data = json.load(f)
                    self.anomaly_samples = set(data.get('anomaly_samples', []))
                logger.info(f"Loaded {len(self.anomaly_samples)} anomaly samples from {anomaly_file}")

                # 加载历史异常样本的详细信息
                self._load_anomaly_details()

            except Exception as e:
                logger.warning(f"Failed to load anomaly samples: {e}")

    def _load_anomaly_details(self):
        """加载历史异常样本的详细信息"""
        info_path = self._get_anomaly_info_path()
        if not info_path:
            return

        # 可以在这里加载额外的历史信息，如果有需要的话
        pass

    def _save_anomaly_samples(self, incremental=False):
        """保存异常样本记录"""
        info_path = self._get_anomaly_info_path()
        if not info_path:
            return

        self._ensure_anomaly_dirs()

        anomaly_file = os.path.join(info_path, self.anomaly_samples_file)

        try:
            if incremental and os.path.exists(anomaly_file):
                # 增量保存：读取现有数据并追加新样本
                with open(anomaly_file, 'r') as f:
                    existing_data = json.load(f)

                # 合并样本哈希
                existing_samples = set(existing_data.get('anomaly_samples', []))
                all_samples = existing_samples.union(self.new_anomaly_samples)

                data = {
                    'anomaly_samples': list(all_samples),
                    'last_updated': time.time(),
                    'total_count': len(all_samples),
                    'update_info': {
                        'new_samples_count': len(self.new_anomaly_samples),
                        'update_time': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                }
            else:
                # 全新保存
                data = {
                    'anomaly_samples': list(self.anomaly_samples),
                    'last_updated': time.time(),
                    'total_count': len(self.anomaly_samples)
                }

            # 保存主文件
            with open(anomaly_file, 'w') as f:
                json.dump(data, f, indent=2)

            # 同时保存备份文件
            backup_file = os.path.join(info_path, f"anomaly_samples_backup_{int(time.time())}.json")
            with open(backup_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(data['anomaly_samples'])} anomaly samples to {anomaly_file}")

        except Exception as e:
            logger.warning(f"Failed to save anomaly samples: {e}")

    def _prepare_batch_for_saving(self, batch):
        """准备batch数据用于保存（处理CUDA tensor）"""

        def process_tensor(tensor):
            if isinstance(tensor, torch.Tensor):
                # 移动到CPU并转换为numpy
                return tensor.detach().cpu().numpy()
            return tensor

        def process_dict(data_dict):
            result = {}
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    result[key] = process_tensor(value)
                elif isinstance(value, dict):
                    result[key] = process_dict(value)
                elif isinstance(value, (list, tuple)):
                    result[key] = [process_tensor(item) if isinstance(item, torch.Tensor) else item for item in value]
                else:
                    result[key] = value
            return result

        if isinstance(batch, (list, tuple)):
            return [process_tensor(item) if isinstance(item, torch.Tensor) else item for item in batch]
        elif isinstance(batch, dict):
            return process_dict(batch)
        else:
            return process_tensor(batch)

    def _generate_sample_hash(self, batch) -> str:
        """生成样本的哈希标识"""
        try:
            input_x, label, other_args = batch
            # 使用输入数据和标签的前几个元素生成哈希
            sample_data = {
                'input_shape': input_x.shape,
                'input_head': input_x.flatten()[:20].tolist() if input_x.numel() > 0 else [],
                'label_shape': label.shape,
                'label_head': label.flatten()[:20].tolist() if label.numel() > 0 else [],
                'tokens_num': other_args.get('tokens_num', 0)
            }
            return str(hash(json.dumps(sample_data, sort_keys=True)))
        except Exception as e:
            logger.warning(f"Failed to generate sample hash: {e}")
            return str(hash(str(time.time())))

    def _is_anomaly_loss(self, current_loss: float) -> bool:
        """检测是否为异常loss"""
        if not self.previous_losses:
            return False

        # 计算最近loss的平均值
        avg_loss = sum(self.previous_losses) / len(self.previous_losses)

        # 如果当前loss与平均值的差异超过阈值，认为是异常
        if avg_loss > 0:  # 避免除零
            ratio = current_loss / avg_loss
            if ratio > self.anomaly_threshold or ratio < 1.0 / self.anomaly_threshold:
                return True

        return False

    def _record_anomaly_sample(self, batch, loss_value: float):
        """记录异常样本"""
        sample_hash = self._generate_sample_hash(batch)

        if sample_hash in self.anomaly_samples:
            return  # 已经记录过的样本不再重复记录

        self.anomaly_samples.add(sample_hash)
        self.new_anomaly_samples.add(sample_hash)

        # 准备异常样本信息
        anomaly_info = {
            'sample_hash': sample_hash,
            'loss': loss_value,
            'step': self.cur_step,
            'epoch': self.cur_epoch,
            'timestamp': time.time(),
            'time_str': time.strftime('%Y-%m-%d %H:%M:%S'),
            'batch_info': {
                'input_shape': batch[0].shape if batch[0] is not None else None,
                'label_shape': batch[1].shape if batch[1] is not None else None,
                'tokens_num': batch[2].get('tokens_num', 0) if len(batch) > 2 else 0
            }
        }

        # 保存详细信息
        info_path = self._get_anomaly_info_path()
        if info_path:
            anomaly_detail_file = os.path.join(info_path, f"anomaly_detail_{sample_hash}.json")
            try:
                with open(anomaly_detail_file, 'w') as f:
                    json.dump(anomaly_info, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save anomaly detail: {e}")

        # 保存原始batch数据
        base_path = self._get_anomaly_base_path()
        if base_path:
            try:
                # 准备数据用于保存
                prepared_batch = self._prepare_batch_for_saving(batch)

                # 保存batch数据
                batch_file = os.path.join(base_path, f"batch_{sample_hash}.pkl")
                with open(batch_file, 'wb') as f:
                    pickle.dump({
                        'batch_data': prepared_batch,
                        'sample_hash': sample_hash,
                        'save_time': time.time()
                    }, f)

            except Exception as e:
                logger.warning(f"Failed to save batch data: {e}")

        logger.warning(f"Anomaly sample detected: loss={loss_value:.4f}, hash={sample_hash}")

        # 定期保存（每记录10个新样本就保存一次）
        if len(self.new_anomaly_samples) % 10 == 0:
            self._save_anomaly_samples(incremental=True)

    def _should_skip_sample(self, batch) -> bool:
        """判断是否应该跳过当前样本"""
        if not self.skip_anomaly_samples:
            return False

        sample_hash = self._generate_sample_hash(batch)
        return sample_hash in self.anomaly_samples

    # 其他方法保持不变（evaluate, start, _start_debug, skip_processed_data, train_step等）
    # 这里省略了其他方法的重复代码，只展示新增和修改的部分

    def evaluate(self):
        eval_loss = 0
        ppl_tokens_num = 0
        ppl = 0
        bar = tqdm.tqdm(total=len(self.eval_dataloader))
        torch.cuda.empty_cache()
        for i, batch in enumerate(self.eval_dataloader):
            try:
                # 检查是否应该跳过异常样本
                if self._should_skip_sample(batch):
                    logger.info(f"Skipping anomaly sample in evaluation")
                    continue

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

    # train_step 方法中需要添加异常检测和记录的代码
    def train_step(self, batch, hvars: Dict = {}):
        """
        Behaviours: training with given batch (forward and backward, update params), normalize loss if it is necessary,
                    do logging of training or evaluating, saving params.
        """
        # 检查是否应该跳过异常样本
        if self._should_skip_sample(batch):
            logger.info(f"Skipping anomaly sample at step {self.cur_step}")
            # 更新计数器但不进行训练
            self.cur_step += 1
            return None, -1

        # ... 原有的train_step代码 ...

        try:
            # ... 原有的forward和loss计算代码 ...

            # 记录loss用于异常检测
            loss_value = loss.item() if hasattr(loss, 'item') else loss
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()

            # 更新loss窗口
            self.previous_losses.append(loss_value)
            if len(self.previous_losses) > self.loss_window_size:
                self.previous_losses.pop(0)

            # 检测异常loss并记录样本
            if len(self.previous_losses) >= 5:  # 至少有5个历史loss值才开始检测
                if self._is_anomaly_loss(loss_value):
                    self._record_anomaly_sample(batch, loss_value)

        # ... 原有的异常处理代码 ...

        finally:
            # ... 原有的finally代码 ...
            pass

        return loss, eval_loss

    # 其他方法保持不变...