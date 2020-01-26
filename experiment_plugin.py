from abc import ABC, abstractmethod


class ExperimentPlugin(ABC):
    @abstractmethod
    def construct(self, model, lr):
        pass

    @abstractmethod
    def sample_step(self, x, y, pred):
        pass

    @abstractmethod
    def epoch_step(self):
        pass