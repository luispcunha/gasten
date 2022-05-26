import wandb


class MetricsLogger:
    def __init__(self, prefix=None, log_epoch=True):
        self.prefix = prefix
        self.step_metrics = []
        self.running_stats = {}
        self.stats = {}
        self.epoch = 0
        self.step = 0
        self.step_counter = 0
        self.log_epoch = log_epoch

    def add_media_log(self, name):
        wandb.define_metric(name, step_metrics=self.apply_prefix('epoch'))

    def log_media(self, name, image, caption=None):
        wandb.log({name: wandb.Image(image, caption=caption)})

    def apply_prefix(self, name):
        return f'{self.prefix}/{name}' if self.prefix is not None else name

    def add(self, name, every_step=False):
        name = self.apply_prefix(name)

        self.stats[name] = []
        wandb.define_metric(name, step_metric=self.apply_prefix('epoch'))

        if every_step:
            self.step_metrics.append(name)
            self.stats[f'{name}_per_step'] = []
            self.running_stats[name] = 0
            wandb.define_metric(
                f'{name}_step', step_metric=self.apply_prefix('step'))

    def reset_step_metrics(self):
        self.step_counter = 0
        for name in self.step_metrics:
            self.running_stats[name] = 0

    def update_step(self, name, avg_value, n_items):
        name = self.apply_prefix(name)

        self.stats[name].append(avg_value)
        self.running_stats[name] += avg_value * n_items

    def update_epoch(self, name, value, prnt=False):
        name = self.apply_prefix(name)

        self.stats[name].append(value)
        wandb.log({name: value}, commit=False)
        if prnt:
            print(name, " = ", value)

    def finalize_step(self, n_step_items=1):
        self.step_counter += n_step_items
        self.step += 1

    def finalize_epoch(self):
        # compute average of step metrics per epoch
        for name in self.step_metrics:
            epoch_value = self.running_stats[name] / self.step_counter
            self.stats[name].append(epoch_value)
            wandb.log({name: epoch_value}, commit=False)

        self.epoch += 1
        wandb.log({self.apply_prefix('epoch'): self.epoch}, commit=True)
