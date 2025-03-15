import numpy as np
import torch


def create_collate_fn_normal_batch(tokenizer, max_tokens=2048, dtype=torch.long):
    def fn(batch):
        b_lengths = [len(i) for i in batch]
        min_b_length = min(b_lengths)
        max_b_length = max(b_lengths)
        # test_max_length = 150
        if max_b_length > max_tokens:  # 2048:
            batch = [i[:max_tokens] for i in batch]
            max_b_length = max_tokens

        bs = len(batch)
        input_ndarray = np.ones((bs, max_b_length)) * tokenizer.pad_id
        for i in range(bs):
            input_ndarray[i, :b_lengths[i]] = batch[i]

        input_x = torch.tensor(input_ndarray, dtype=dtype)  # .to(self.device)
        input_y = input_x[:, 1:]
        input_y = input_y  # .flatten()

        return [
            input_x, input_y,
            {
                'start_pos': 0,
                'tokens_num': sum([len(i) for i in batch])
            }
        ]

    return fn
