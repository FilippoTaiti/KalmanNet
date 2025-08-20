import numpy as np
import torch

class EarlyStop:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best = None
        self.count = 0
        self.stop = False


    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best is None:
            self.best = score
            self.save(val_loss, model)
        elif score < self.best + self.delta:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True
        else:
            self.best = score
            self.count = 0
            self.save(val_loss, model)


    def save(self, val_loss, model):
        torch.save(model.state_dict(), 'KNet/checkpoint.pt')
