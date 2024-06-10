import math
from torch.optim.optimizer import Optimizer#
import warnings
from torch.optim.lr_scheduler import _LRScheduler
from typing import Union, List, Optional

class CosineAnnealingWithWarmRestarts(_LRScheduler):
    def __init__(
        self, 
        optimizer: Optimizer,
        cycle_period: Union[float, int],
        cycle_mult: float = 1,
        warmup_period: Union[float, int] = 0,
        warmup_once: bool = False,
        max_lr: Optional[Union[float, List[float]]] = None, 
        min_lr: float = 1e-8,
        gamma: float = 1,
        strategy: str = 'step',
    ):
        """Cosine annealing with warm restarts scheduler. Implements the cosine annealing scheduler with warm restarts from the paper SGDR (https://arxiv.org/abs/1608.03983).

        Args:
            optimizer (Optimizer): PyTorch optimizer
            cycle_period (Union[float, int]): The period for the first cycle. If strategy is 'step', this is the number of steps in the first cycle. 
                                If strategy is 'epoch', this is the number of epochs in the first cycle.
            cycle_mult (float): The multiplier for the cycle period after each cycle. Defaults to 1.
            warmup_period (Union[float, int]): The period for warmup for each cycle. 
                                               If strategy is 'step', this is the number of steps for the warmup. 
                                               If strategy is 'epoch', this is the number of epochs for the warmup. 
                                               Defaults to 0.
            warmup_once (bool): Whether to apply warmup only once at the beginning of the first cycle. Only affects when warmup_period > 0. Defaults to False.
            max_lr (Union[float, List[float]], optional): The maximum learning rate for the optimizer (eta_max). 
                                                          If ommited, the learning rate of the optimizer will be used. 
                                                          If a float is given, all lr in the optimizer param groups will be overriden with this value. 
                                                          If a list is given, the length of the list must be the same as the number of param groups in the optimizer.
                                                          Defaults to None.
            min_lr (float, optional): The minimum learning rate for the optimizer (eta_min). Defaults to 1e-8.
            gamma (float, optional): The decay rate for the learning rate after each cycle. Defaults to 1.
            strategy (str, optional): Defines whether the cycle period and warmup period to be treated as steps or epochs. Can be `step` or `epoch`. 
                                      Note that if you use `epoch`, you need to specify the epoch progress each time you call `.step()` Defaults to 'step'.
        """
        self.cycle_period = cycle_period
        self.cycle_mult = cycle_mult
        self.warmup_period = warmup_period
        self.warmup_once = warmup_once
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.gamma = gamma
        self.strategy = strategy
        
        self.cur_cycle_period = cycle_period
        self.cur_step = 0
        self.first_step = True
        self.first_cycle_completed = False
        if strategy == 'epoch':
            self.cycle_reducer = 0

        assert strategy in ['step', 'epoch'], 'strategy must be either `step` or `epoch`'
        assert cycle_mult >= 1, 'cycle_mult must be greater than or equal to 1'
        assert 0 <= warmup_period < cycle_period, 'warmup_period must be greater than or equal to 0 and less than cycle_period'

        if max_lr is not None:
            if isinstance(max_lr, list):
                assert len(max_lr) == len(optimizer.param_groups), 'max_lr must be a list of the same length as optimizer.param_groups'
            else:
                max_lr = [max_lr] * len(optimizer.param_groups)
            
            for group, lr in zip(optimizer.param_groups, max_lr):
                group['lr'] = lr
                group['initial_lr'] = lr

        super().__init__(optimizer, -1)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer')}
        return state_dict

    def load_state_dict(self, state_dict):
        if self.strategy == 'epoch':
            warnings.warn('Restoring scheduler state with `epoch` strategy may lead to unexpected behavior.')
        super().load_state_dict(state_dict)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn('To get the last learning rate computed by the scheduler, please use `get_last_lr()`.')
        if self.cur_step < self.warmup_period and not (self.warmup_once and self.first_cycle_completed):
            lrs = [self.min_lr + (max_lr - self.min_lr) * self.cur_step / self.warmup_period for max_lr in self.base_lrs]
        else:
            warmup_period = self.warmup_period if not (self.warmup_once and self.first_cycle_completed) else 0
            lrs = [self.min_lr + 0.5 * (max_lr - self.min_lr) * (1 + math.cos(math.pi * (self.cur_step - warmup_period) / (self.cur_cycle_period - warmup_period))) for max_lr in self.base_lrs]
        
        return lrs

    def step(self, epoch:float = None):
        """Step the scheduler to update the learning rate. If strategy is `epoch`, you need to specify the epoch progress each time you call this method.

        Args:
            epoch (float, optional): If strategy is `epoch`, you need to specify the epoch progress each time you call this method. Defaults to None.
        """
        if self.strategy == 'epoch':
            if self.first_step:
                self.first_step = False
                self.cur_step = 0
            else:
                assert epoch is not None, 'You need to specify the epoch progress when using `epoch` strategy.'
                self.cur_step = epoch - self.cycle_reducer
        else:
            if epoch is not None:
                warnings.warn('You specified the epoch progress but the strategy is `step`. The epoch progress will be ignored.')
        if self.cur_step >= self.cur_cycle_period:
            self.first_cycle_completed = True
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
