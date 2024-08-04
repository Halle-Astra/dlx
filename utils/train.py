import torch
import os
from loguru import logger


def save_parameters(save_folder, model, optimizer=None, others=None):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    names = ['model', 'optim', 'others']
    for i, item in enumerate([model, optimizer, others]):
        if item is not None:
            path = os.path.join(save_folder, names[i]+'.pth')
            torch.save(item, path)
            logger.info(f'saved: {path}')
