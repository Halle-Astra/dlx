import torch
import os
from loguru import logger
import json, time, pickle
from typing import Dict, List, Optional, Set, Any


def save_parameters(save_folder, model, optimizer=None, others=None):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    names = ['model', 'optim', 'others']
    for i, item in enumerate([model, optimizer, others]):
        if item is not None:
            path = os.path.join(save_folder, names[i] + '.pth')
            torch.save(item, path)
            logger.info(f'saved: {path}')


class AnomalyRecorder:
    def __init__(self):
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

