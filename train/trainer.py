from dlx.utils.train import save_parameters


class BaseTrainer:
    def __init__(self):
        self.cur_epoch = 0
        self.cur_step = 0
        self.save_folder = None

    def save(self, train_loss=-1, eval_loss=-1):
        folder_name = f'epoch:{self.cur_epoch}-step:{self.cur_step}-train_loss:{train_loss}-eval_loss:{eval_loss}'
        folder = os.path.join(self.save_folder, folder_name)
        save_parameters(
            folder,
            self.model.state_dict(),
            self.optimizer.state_dict(),
        )
