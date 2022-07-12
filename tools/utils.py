import torch

class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()
        self.total_loss = 0
        self.step_count = 0

    def add(self, v):
        value = v.item()
        self.total_loss += value
        self.step_count += 1
    
    def out(self):
        avg_loss = self.total_loss / self.step_count
        self.total_loss = 0
        self.step_count = 0
        return avg_loss

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res