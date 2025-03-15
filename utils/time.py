import torch
import time


class Timer:
    def __init__(self):
        from loguru import logger
        is_debug = logger._core.handlers[0].levelno == 10

        def nothing():
            return

        self.maybe_sync = torch.cuda.synchronize if is_debug else nothing

    # @staticmethod
    def mark(self):
        """This method will make your torch project slower than normal."""
        self.maybe_sync()
        # torch.cuda.synchronize()
        return time.time()


timer = Timer()
