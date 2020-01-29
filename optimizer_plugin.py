from experiment_plugin import ExperimentPlugin


class OptimizerPlugin(ExperimentPlugin):
    def __init__(self, make_optimizer, make_scheduler):
        self.make_optimizer = make_optimizer
        self.make_scheduler = make_scheduler
        self.optimizer = None
        self.scheduler = None

    def construct(self, model, lr):
        self.optimizer = self.make_optimizer(model, lr)
        self.scheduler = self.make_scheduler(self.optimizer, lr)

    def process_sample(self, x, y, pred):
        self.optimizer.step()

    def end_epoch(self):
        self.scheduler.step()
