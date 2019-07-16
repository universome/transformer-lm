"NoamOpt. Taken from https://github.com/harvardnlp/annotated-transformer"
from torch.optim import Adam


class TriangleAdam:
    "Optim wrapper that makes Adam triangle."
    def __init__(self, parameters, config):
        lr = config.get('lr', 1e-3)
        betas = config.get('betas', (0.9, 0.999))

        self.optimizer = Adam(parameters, lr=lr, betas=betas)
        self._step = 0
        self.warmup_steps = config.warmup
        self.factor = config.factor
        self.model_size = config.model_size
        self._current_lr = 0

    def step(self):
        "Update parameters and lr"
        self._step += 1
        lr = self.compute_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self._current_lr = lr
        self.optimizer.step()

    def compute_lr(self, step=None):
        step = step or self._step

        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()
