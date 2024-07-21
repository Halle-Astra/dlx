class Llama3Trainer:
    def __init__(self, model, dataloader, loss_modules):
        self.model = model
        self.dataloader = dataloader
        if not isinstance(loss_modules, list):
            loss_modules = [loss_modules]
        self.loss_modules = loss_modules

    def start(self):
        for batch in self.dataloader:
            x = batch
            output = self.model(batch)
            loss = 0
            for loss_m in self.loss_modules:
                loss += loss_m(x, output)

