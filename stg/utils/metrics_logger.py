from torch import log_
import wandb
import matplotlib.pyplot as plt


class MetricsLogger:
    def __init__(self, prefix=None, log_epoch=True):
        self.prefix = prefix
        self.iteration_metrics = []
        self.running_stats = {}
        self.it_counter = {}
        self.stats = {}
        self.epoch = 0
        self.log_epoch = log_epoch

    def add_media_metric(self, name):
        wandb.define_metric(name, step_metric=self.apply_prefix('epoch'))

    def log_image(self, name, image, caption=None):
        wandb.log({name: wandb.Image(image, caption=caption)})

    def log_plot(self, name):
        wandb.log({name: plt})

    def apply_prefix(self, name):
        return f'{self.prefix}/{name}' if self.prefix is not None else name

    def add(self, name, iteration_metric=False):
        self.stats[name] = []
        wandb.define_metric(self.apply_prefix(name),
                            step_metric=self.apply_prefix('epoch'))

        if iteration_metric:
            self.iteration_metrics.append(name)
            self.stats[f'{name}_per_it'] = []
            self.running_stats[name] = 0
            self.it_counter[name] = 0

    def reset_it_metrics(self):
        for name in self.step_metrics:
            self.running_stats[name] = 0
            self.it_counter[name] = 0

    def update_it_metric(self, name, value):
        self.running_stats[name] += value
        self.it_counter[name] += 1

    def update_epoch_metric(self, name, value, prnt=False):
        self.stats[name].append(value)
        wandb.log({self.apply_prefix(name): value}, commit=False)

        if prnt:
            print(name, " = ", value)

    def finalize_epoch(self):
        # compute average of iteration metrics per epoch
        for name in self.iteration_metrics:
            epoch_value = self.running_stats[name] / self.it_counter[name]
            self.stats[name].append(epoch_value)
            wandb.log({self.apply_prefix(name): epoch_value}, commit=False)

            if self.log_epoch:
                print(name, " = ", epoch_value)

        self.epoch += 1
        wandb.log({self.apply_prefix('epoch'): self.epoch}, commit=True)
