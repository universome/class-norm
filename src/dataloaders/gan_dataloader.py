import torch.nn as nn

class GANDataloader:
    """
    Dataloader wrapper for a generator, which samples data from it
    until max_num_iters is reached
    """
    def __init__(self, sample_fn, max_num_iters:int):
        self.sample_fn = sample_fn
        self.max_num_iters = max_num_iters
        self._curr_iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_iter > self.max_num_iters:
            raise StopIteration
        else:
            self._curr_iter += 1

            return self.sample_fn()

    def __len__(self) -> int:
        return self.max_num_iters
