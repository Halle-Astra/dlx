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
                 kv_cache_enabled=False,
                 device='cuda',
                 dtype=torch.float16,
                 parallel=None):
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
            start_pos_to_wait_predict = random.randint(1, min_b_length - 1)  # 不能输入空字符串

            bs = len(batch)
            input_tensor = np.ones((bs, max_b_length)) * self.tokenizer.pad_id
            for i in range(bs):
                input_tensor[i, :b_lengths[i]] = batch[i]

            start_index = 0
            for end_index in range(start_pos_to_wait_predict, max_b_length - 1):
                input_x = input_tensor[:, start_index: end_index]
                input_list = []
                label_list = []
                for i in range(bs):
                    if not bool(input_tensor[i][end_index] == self.tokenizer.pad_id):
                        input_list.append(input_x[i])
                        label_list.append(input_tensor[i][end_index])

                input_x = torch.tensor(np.vstack(input_list), dtype=self.dtype).to(self.device)
                input_y = torch.tensor(label_list, dtype=self.dtype).to(self.device)

                output = self.model(input_x, start_index)
                output = output[:, -1]
                print(output.detach().numpy())
                # output_tid = torch.argmax(output, dim=-1)
                loss = 0
                for loss_m in self.loss_modules:
                    loss += loss_m(output, input_y)
                if not torch.isnan(loss):
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                else:
                    print('nan ------------------')

                start_index = end_index

                print(loss.item())
