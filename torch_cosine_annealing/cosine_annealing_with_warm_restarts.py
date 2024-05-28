import math
from torch.optim.optimizer import Optimizer
import warnings
from torch.optim.lr_scheduler import _LRScheduler
from typing import Union, List

class CosineAnnealingWithWarmRestarts(_LRScheduler):
    def __init__(
        self, 
        optimizer: Optimizer,
        cycle_period: int,
        cycle_mult: int,
        warmup_period: Union[float, int] = 0,
        max_lr: Union[float, List[float]] = None, 
        min_lr: float = 1e-8,
        gamma: float = 1,
        strategy: str = 'step',
        last_epoch: int = -1
    ) -> None:
        self.cycle_period = cycle_period
        self.warmup_period = warmup_period
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cur_cycle_period = cycle_period
        self.gamma = gamma
        self.cur_step = 0
        self.strategy = strategy
        self.first_step = True
        if strategy == 'epoch':
            self.cycle_reducer = 0

        super().__init__(optimizer, last_epoch)

        if max_lr is not None:
            if isinstance(max_lr, list):
                assert len(max_lr) == len(optimizer.param_groups), 'max_lr must be a list of the same length as optimizer.param_groups'
            else:
                max_lr = [max_lr] * len(optimizer.param_groups)
            
            if last_epoch == -1:
                for group, lr in zip(optimizer.param_groups, max_lr):
                    group['lr'] = lr
                    group.setdefault('initial_lr', group['lr'])
            else:
                for i, group in enumerate(optimizer.param_groups):
                    if 'initial_lr' not in group:
                        raise KeyError('param `initial_lr` is not specified in param_groups[{i}] when resuming an optimizer')
        
            self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]


    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer')}
        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        if self.strategy == 'epoch':
            warnings.warn('Restoring scheduler state with epoch strategy may lead to unexpected behavior.')

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn('To get the last learning rate computed by the scheduler, please use `get_last_lr()`.')
        if self.cur_step < self.warmup_period:
            lrs = [self.min_lr + (max_lr - self.min_lr) * self.cur_step / self.warmup_period for max_lr in self.base_lrs]
        else:
            lrs = [self.min_lr + 0.5 * (max_lr - self.min_lr) * (1 + math.cos(math.pi * (self.cur_step - self.warmup_period) / (self.cur_cycle_period - self.warmup_period))) for max_lr in self.base_lrs]
        
        return lrs

    def step(self, epoch:float = None):
        if self.strategy == 'epoch':
            if self.first_step:
                self.first_step = False
                self.cur_step = 0
            else:
                self.cur_step = epoch - self.cycle_reducer

        if self.cur_step > self.cur_cycle_period:
            if self.strategy == 'epoch':
                self.cycle_reducer += self.cur_cycle_period
            self.cur_step = 0
            self.cur_cycle_period = self.cur_cycle_period * self.cycle_mult
            self.base_lrs = [lr * self.gamma for lr in self.base_lrs]

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self
        
        with _enable_get_lr_call(self):
            lrs = self.get_lr()
        
        if self.strategy == 'step':
            self.cur_step += 1

        for i in range(len(lrs)):
            self.optimizer.param_groups[i]['lr'] = lrs[i]
        
        self._last_lr = lrs
