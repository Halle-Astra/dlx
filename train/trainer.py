from dlx.utils.train import save_parameters
import os
import glob
import torch
from torch.nn.parallel import DistributedDataParallel as DDP



class BaseTrainer:
    def __init__(self):
        self.cur_epoch = 0
        self.cur_step = 0
        self.save_folder = None

    def save(self, train_loss=-1, eval_loss=-1):
        folder_name = f'epoch:{self.cur_epoch}-step:{self.cur_step}-train_loss:{train_loss}-eval_loss:{eval_loss}'
        folder = os.path.join(self.save_folder, folder_name)
        others = dict(cur_step=self.cur_step,
                      cur_epoch=self.cur_epoch,
                      loss=train_loss,
                      eval_loss=eval_loss)
        save_parameters(
            folder,
            self.model.state_dict(),
            self.optimizer.state_dict(),
            others
        )

    def load_weights(self, weights_path, prefix='', postfix='pth'):
        if os.path.isfile(weights_path):
            weights = [weights_path]
        else:
            file_format = '*'.join([prefix, postfix])
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
                self.model.load_state_dict(weight)

    def resume(self, folder):
        if 'latest' in os.listdir(folder):
            folder = os.path.join(folder, 'latest')
            assert os.path.isdir(folder), 'Argument folder should be a directory.'

        model_path = os.path.join(folder, 'model.pth')
        others_path = os.path.join(folder, 'others.pth')
        optim_path = os.path.join(folder, 'optim.pth')

        self.load_weights(model_path)

        optim_weights = torch.load(optim_path)
        self.optimizer.load_state_dict(optim_weights)

        others = torch.load(others_path)
        self.cur_step = others['cur_step']
        self.cur_epoch = others['cur_epoch']



