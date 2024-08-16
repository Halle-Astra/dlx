import torch
import time


class Timer:
    def __init__(self):
        pass

    @staticmethod
    def mark():
        """This method will make your torch project slower than normal."""
        torch.cuda.synchronize()
        return time.time()


timer = Timer()
