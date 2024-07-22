import random



class AutoRegressiveTrainer:
    def __init__(self, model, dataloader, loss_modules, model_with_kv_cache=False):
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

    def start(self):
        for batch in self.dataloader:
            b_lengths = [len(i) for i in batch]
            min_b_length = min(b_lengths)
            start_site = random.randint(0, min_b_length-1)

            output = self.model(batch)
            loss = 0
            for loss_m in self.loss_modules:
                loss += loss_m(x, output)

