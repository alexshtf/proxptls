import torch

from experiment_plugin import ExperimentPlugin


class ProxPointPlugin(ExperimentPlugin):
    def __init__(self):
        self.model = None
        self.lr = None

    def construct(self, model, lr):
        self.model = model
        self.lr = lr

    def process_sample(self, x, y, pred):
        with torch.no_grad():
            numerator = self.lr * (pred - y)
            denominator = (1 + self.lr * (1 + torch.dot(x, x)))
            coeff = numerator / denominator

            self.model.beta -= coeff * x
            self.model.alpha -= coeff

    def end_epoch(self):
        pass

