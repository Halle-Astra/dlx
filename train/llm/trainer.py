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


class AutoRegressiveTrainer:
    def __init__(self, model, dataloader,
                 loss_modules,
                 optimizer=None,
                 world_size=None,
                 tokenizer=None,
                 model_is_kv_cache_enabled=False,
                 device='cuda',
                 dtype=torch.float16,
                 parallel=None,
                 grad_clip=None):
        """

        :param model:
        :param dataloader:          A dataloader which only generate a batch of list of token ids
        :param loss_modules:
        :param kv_cache_enabled:    determine the training strategy, like GPT if false, or like Llama3 if true, default
                                    value is false.
        """
        # if parallel is not None and parallel == 'ddp':
        #     self.init_parallel()

        self.model = model
        self.dataloader = dataloader
        if not isinstance(loss_modules, list):
            loss_modules = [loss_modules]
        self.loss_modules = loss_modules
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.world_size = world_size
        self.grad_clip = grad_clip
        self.model_is_kv_cache_enabled = model_is_kv_cache_enabled

    def init_parallel(self):
        torch.cuda.set_device(dist.get_rank())
        model_parallel_size = self.world_size
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

    def start(self):
        for batch in self.dataloader:
            b_lengths = [len(i) for i in batch]
            min_b_length = min(b_lengths)
            max_b_length = max(b_lengths)
            if max_b_length > 2048:
                batch = [i[:2048] for i in batch]
                max_b_length = 2048
            start_pos_to_wait_predict = random.randint(1, min_b_length - 1)  # 不能输入空字符串

            bs = len(batch)
            input_ndarray = np.ones((bs, max_b_length)) * self.tokenizer.pad_id
            for i in range(bs):
                input_ndarray[i, :b_lengths[i]] = batch[i]

            loss = 0
            start_index = 0
            # t = 0
            # for end_index in range(start_pos_to_wait_predict, max_b_length - 1):
            #     # t += 1
            #     if end_index > 400:  # or t > 30:
            #         break
            #     input_x = input_tensor[:, start_index: end_index]
            #     input_list = []
            #     label_list = []
            #     index_in_batch = []
            #     for i in range(bs):
            #         if not bool(input_tensor[i][end_index] == self.tokenizer.pad_id):
            #             input_list.append(input_x[i])
            #             label_list.append(input_tensor[i][end_index])
            #             index_in_batch.append(i)

            input_x = torch.tensor(input_ndarray, dtype=self.dtype).to(self.device)
            input_y = input_x[1:]

            output = self.model(input_x, start_index)
            output = output[:, :-1]
            print(output.detach().cpu().numpy())
            # output_tid = torch.argmax(output, dim=-1)
            for loss_m in self.loss_modules:
                loss_item = loss_m(output, input_y)
                if not torch.isnan(loss_item):
                    loss = loss + loss_item

            start_index = end_index

            print(loss.item())
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

    def train_with_saving_memory(self):
        """
        This method is written by mimic Llama3 inference code, but discarded now.
        :return:
        """
        for batch in self.dataloader:
            b_lengths = [len(i) for i in batch]
            min_b_length = min(b_lengths)
            max_b_length = max(b_lengths)
            start_pos_to_wait_predict = random.randint(1, min_b_length - 1)  # 不能输入空字符串

            bs = len(batch)
            input_tensor = np.ones((bs, max_b_length)) * self.tokenizer.pad_id
            for i in range(bs):
                input_tensor[i, :b_lengths[i]] = batch[i]

            loss = 0
            start_index = 0
            # t = 0
            for end_index in range(start_pos_to_wait_predict, max_b_length - 1):
                # t += 1
                if end_index > 400:  # or t > 30:
                    break
                input_x = input_tensor[:, start_index: end_index]
                input_list = []
                label_list = []
                index_in_batch = []
                for i in range(bs):
                    if not bool(input_tensor[i][end_index] == self.tokenizer.pad_id):
                        input_list.append(input_x[i])
                        label_list.append(input_tensor[i][end_index])
                        index_in_batch.append(i)

                input_x = torch.tensor(np.vstack(input_list), dtype=self.dtype).to(self.device)
                input_y = torch.tensor(label_list, dtype=self.dtype).to(self.device)

                output = self.model(input_x, start_index, index_in_batch)
                output = output[:, -1]
                print(output.detach().cpu().numpy())
                # output_tid = torch.argmax(output, dim=-1)
                for loss_m in self.loss_modules:
                    loss_item = loss_m(output, input_y)
                    if not torch.isnan(loss_item):
                        loss = loss + loss_item

                start_index = end_index

                print(loss.item())
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
