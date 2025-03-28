from dlx.utils.train import save_parameters
import os
import glob
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger


class BaseTrainer:
    def __init__(self, ):
        self.cur_epoch = 0
        self.cur_step = 0
        self.tokens_num = 0
        self.save_folder = None
        self.accumulate_iters = 1

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
