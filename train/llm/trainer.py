import random
import numpy as np


class AutoRegressiveTrainer:
    def __init__(self, model, dataloader, loss_modules, tokenizer=None, model_with_kv_cache=False, device='cuda',
                 dtype=None):
        """

        :param model:
        :param dataloader:          A dataloader which only generate a batch of list of token ids
        :param loss_modules:
        :param model_with_kv_cache: determine the training strategy, like GPT if false, or like Llama3 if true, default
                                    value is false.
        """
        self.model = model
        self.dataloader = dataloader
        if not isinstance(loss_modules, list):
            loss_modules = [loss_modules]
        self.loss_modules = loss_modules
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

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
                    if np.any(input_x[i] == self.tokenizer.pad_id):
                        input_list.append(input_x[i])
                        label_list.append(input_x[i])

            output = self.model(batch)
            loss = 0
            for loss_m in self.loss_modules:
                loss += loss_m(x, output)
