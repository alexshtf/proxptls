from abc import ABC, abstractmethod


class ExperimentPlugin(ABC):
    @abstractmethod
    def construct(self, model, lr):
        pass

    @abstractmethod
    def process_sample(self, x, y, pred):
        pass

    @abstractmethod
    def end_epoch(self):
        pass