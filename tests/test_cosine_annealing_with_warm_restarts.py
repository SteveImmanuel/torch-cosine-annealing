import pytest
from torch.optim import Optimizer
from torch_cosine_annealing import CosineAnnealingWithWarmRestarts

class MockOptimizer(Optimizer):
    def __init__(self, lrs):
        self.param_groups = []
        for lrs in lrs:
            self.param_groups.append({'lr': lrs})

@pytest.fixture(scope='function')
def optimizer():
    return MockOptimizer([1e-3, 1e-4, 1e-5])

def test_init_step(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=10, 
        cycle_mult=2, 
        warmup_period=2, 
        max_lr=0.1, 
        min_lr=0.01, 
        gamma=0.9, 
        strategy='step'
    )
    assert 'initial_lr' in optimizer.param_groups[0]
    assert scheduler.base_lrs == [1e-1, 1e-1, 1e-1]
    assert optimizer.param_groups[0]['initial_lr'] == 1e-1
    assert optimizer.param_groups[1]['initial_lr'] == 1e-1
    assert optimizer.param_groups[2]['initial_lr'] == 1e-1
    assert not hasattr(scheduler, 'cycle_reducer')

def test_init_epoch(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=5, 
        cycle_mult=2, 
        warmup_period=2, 
        max_lr=0.1, 
        min_lr=0.01, 
        gamma=0.9, 
        strategy='epoch'
    )
    assert 'initial_lr' in optimizer.param_groups[0]
    assert scheduler.base_lrs == [1e-1, 1e-1, 1e-1]
    assert optimizer.param_groups[0]['initial_lr'] == 1e-1
    assert optimizer.param_groups[1]['initial_lr'] == 1e-1
    assert optimizer.param_groups[2]['initial_lr'] == 1e-1
    assert hasattr(scheduler, 'cycle_reducer')

def test_init_wrong_strategy(optimizer: MockOptimizer):
    with pytest.raises(AssertionError) as e:
        scheduler = CosineAnnealingWithWarmRestarts(
            optimizer, 
            cycle_period=5, 
            cycle_mult=2, 
            warmup_period=2, 
            max_lr=0.1, 
            min_lr=0.01, 
            gamma=0.9, 
            strategy='wrong'
        )
    assert str(e.value) == 'strategy must be either `step` or `epoch`'

def test_init_invalid_cycle_mult(optimizer: MockOptimizer):
    with pytest.raises(AssertionError) as e:
        scheduler = CosineAnnealingWithWarmRestarts(
            optimizer, 
            cycle_period=5, 
            cycle_mult=0, 
            warmup_period=2, 
            max_lr=0.1, 
            min_lr=0.01, 
            gamma=0.9, 
            strategy='step'
        )
    assert str(e.value) == 'cycle_mult must be greater than or equal to 1'

def test_init_invalid_warmup_period(optimizer: MockOptimizer):
    with pytest.raises(AssertionError) as e:
        scheduler = CosineAnnealingWithWarmRestarts(
            optimizer, 
            cycle_period=5, 
            cycle_mult=2, 
            warmup_period=5, 
            max_lr=0.1, 
            min_lr=0.01, 
            gamma=0.9, 
            strategy='step'
        )
    assert str(e.value) == 'warmup_period must be greater than or equal to 0 and less than cycle_period'
    with pytest.raises(AssertionError) as e:
        scheduler = CosineAnnealingWithWarmRestarts(
            optimizer, 
            cycle_period=5, 
            cycle_mult=2, 
            warmup_period=-1, 
            max_lr=0.1, 
            min_lr=0.01, 
            gamma=0.9, 
            strategy='step'
        )
    assert str(e.value) == 'warmup_period must be greater than or equal to 0 and less than cycle_period'

def test_max_lr_dont_override(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=5, 
        cycle_mult=2, 
        warmup_period=2, 
        min_lr=0.01, 
        gamma=0.9, 
        strategy='step'
    )
    assert scheduler.base_lrs == [1e-3, 1e-4, 1e-5]
    assert optimizer.param_groups[0]['initial_lr'] == 1e-3
    assert optimizer.param_groups[1]['initial_lr'] == 1e-4
    assert optimizer.param_groups[2]['initial_lr'] == 1e-5

def test_max_lr_override_list(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=5, 
        cycle_mult=2, 
        warmup_period=2, 
        max_lr=[5, 10, 15],
        min_lr=0.01, 
        gamma=0.9, 
        strategy='step'
    )
    assert scheduler.base_lrs == [5, 10, 15]
    assert optimizer.param_groups[0]['initial_lr'] == 5
    assert optimizer.param_groups[1]['initial_lr'] == 10
    assert optimizer.param_groups[2]['initial_lr'] == 15

def test_max_lr_override_list_wrong_param(optimizer: MockOptimizer):
    with pytest.raises(AssertionError) as e:
        scheduler = CosineAnnealingWithWarmRestarts(
            optimizer, 
            cycle_period=5, 
            cycle_mult=2, 
            warmup_period=2, 
            max_lr=[0.1, 0.2],
            min_lr=0.01, 
            gamma=0.9, 
            strategy='step'
        )
    assert str(e.value) == 'max_lr must be a list of the same length as optimizer.param_groups'

def test_state_dict(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=5, 
        cycle_mult=2, 
        warmup_period=2, 
        max_lr=0.1, 
        min_lr=0.01, 
        gamma=0.9, 
        strategy='step'
    )
    state_dict = scheduler.state_dict()
    assert 'optimizer' not in state_dict
    assert state_dict['cycle_period'] == 5
    assert state_dict['cycle_mult'] == 2
    assert state_dict['warmup_period'] == 2
    assert state_dict['max_lr'] == 0.1
    assert state_dict['min_lr'] == 0.01
    assert state_dict['gamma'] == 0.9
    assert state_dict['strategy'] == 'step'
    
def test_load_state_dict(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=5, 
        cycle_mult=2, 
        warmup_period=2, 
        max_lr=0.1, 
        min_lr=0.01, 
        gamma=0.9, 
        strategy='step'
    )
    state_dict = scheduler.state_dict()
    state_dict['cycle_period'] = 10
    state_dict['cycle_mult'] = 3
    state_dict['warmup_period'] = 3
    scheduler.load_state_dict(state_dict)
    assert scheduler.cycle_period == 10
    assert scheduler.cycle_mult == 3
    assert scheduler.warmup_period == 3

def test_load_state_dict_epoch(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=5, 
        cycle_mult=2, 
        warmup_period=2, 
        max_lr=0.1, 
        min_lr=0.01, 
        gamma=0.9, 
        strategy='epoch'
    )
    state_dict = scheduler.state_dict()
    with pytest.warns(Warning) as w:
        scheduler.load_state_dict(state_dict)
    assert w[0].message.args[0] == 'Restoring scheduler state with `epoch` strategy may lead to unexpected behavior.'

def test_get_lr_outside_step(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=5, 
        cycle_mult=2, 
        warmup_period=2, 
        max_lr=0.1, 
        min_lr=0.01, 
        gamma=0.9, 
        strategy='epoch'
    )
    with pytest.warns(Warning) as w:
        lrs = scheduler.get_lr()
    assert w[0].message.args[0] == 'To get the last learning rate computed by the scheduler, please use `get_last_lr()`.'

def test_scheduler_step_warning(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=5, 
        cycle_mult=2, 
        warmup_period=2, 
        max_lr=0.1, 
        min_lr=0.01, 
        gamma=0.9, 
        strategy='step'
    )
    with pytest.warns(Warning) as w:
        scheduler.step(5)
    assert w[0].message.args[0] == 'You specified the epoch progress but the strategy is `step`. The epoch progress will be ignored.'

def test_scheduler_step_no_warmup(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=25, 
        cycle_mult=1, 
        warmup_period=0, 
        max_lr=[100, 200, 300], 
        min_lr=10, 
        gamma=1, 
        strategy='step'
    )
    lrs = []
    n_steps = 100
    for _ in range(n_steps):
        lrs.append(scheduler.get_last_lr())
        scheduler.step()
    
    assert lrs[0][0] == 100
    assert lrs[0][1] == 200
    assert lrs[0][2] == 300
    assert lrs[50][0] == 100
    assert lrs[50][1] == 200
    assert lrs[50][2] == 300

def test_scheduler_step_cycle_mult(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=25, 
        cycle_mult=2, 
        warmup_period=0, 
        max_lr=[100, 200, 300], 
        min_lr=10, 
        gamma=1, 
        strategy='step'
    )
    lrs = []
    n_steps = 100
    for _ in range(n_steps):
        lrs.append(scheduler.get_last_lr())
        scheduler.step()
    
    assert lrs[0][0] == 100
    assert lrs[0][1] == 200
    assert lrs[0][2] == 300
    assert lrs[75][0] == 100
    assert lrs[75][1] == 200
    assert lrs[75][2] == 300

def test_scheduler_step_gamma(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=25, 
        cycle_mult=1, 
        warmup_period=0, 
        max_lr=[100, 200, 300], 
        min_lr=10, 
        gamma=0.5, 
        strategy='step'
    )
    lrs = []
    n_steps = 100
    for _ in range(n_steps):
        lrs.append(scheduler.get_last_lr())
        scheduler.step()
    
    assert lrs[0][0] == 100
    assert lrs[0][1] == 200
    assert lrs[0][2] == 300
    assert lrs[50][0] == 25
    assert lrs[50][1] == 50
    assert lrs[50][2] == 75

def test_scheduler_step_warmup(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=25, 
        cycle_mult=1, 
        warmup_period=5, 
        max_lr=[100, 200, 300], 
        min_lr=10, 
        gamma=1, 
        strategy='step'
    )
    lrs = []
    n_steps = 100
    for _ in range(n_steps):
        lrs.append(scheduler.get_last_lr())
        scheduler.step()
    
    assert lrs[0][0] == 10
    assert lrs[0][1] == 10
    assert lrs[0][2] == 10
    assert lrs[25][0] == 10
    assert lrs[25][1] == 10
    assert lrs[25][2] == 10
    assert lrs[55][0] == 100
    assert lrs[55][1] == 200
    assert lrs[55][2] == 300

def test_scheduler_step_warmup_once(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=25, 
        cycle_mult=1, 
        warmup_period=5, 
        warmup_once=True,
        max_lr=[100, 200, 300], 
        min_lr=10, 
        gamma=1, 
        strategy='step'
    )
    lrs = []
    n_steps = 100
    for _ in range(n_steps):
        lrs.append(scheduler.get_last_lr())
        scheduler.step()
    
    assert lrs[0][0] == 10
    assert lrs[0][1] == 10
    assert lrs[0][2] == 10
    assert lrs[5][0] == 100
    assert lrs[5][1] == 200
    assert lrs[5][2] == 300
    assert lrs[75][0] == 100
    assert lrs[75][1] == 200
    assert lrs[75][2] == 300

def test_scheduler_step_combination(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=25, 
        cycle_mult=2, 
        warmup_period=5, 
        max_lr=[100, 200, 300], 
        min_lr=10, 
        gamma=0.5, 
        warmup_once=True,
        strategy='step'
    )
    lrs = []
    n_steps = 100
    for _ in range(n_steps):
        lrs.append(scheduler.get_last_lr())
        scheduler.step()
    
    assert lrs[0][0] == 10
    assert lrs[0][1] == 10
    assert lrs[0][2] == 10
    assert lrs[5][0] == 100
    assert lrs[5][1] == 200
    assert lrs[5][2] == 300
    assert lrs[75][0] == 25
    assert lrs[75][1] == 50
    assert lrs[75][2] == 75

def test_scheduler_epoch_step_fail(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=5, 
        cycle_mult=2, 
        warmup_period=2, 
        max_lr=0.1, 
        min_lr=0.01, 
        gamma=0.9, 
        strategy='epoch'
    )
    with pytest.raises(AssertionError) as e:
        scheduler.step()
    assert str(e.value) == 'You need to specify the epoch progress when using `epoch` strategy.'

def test_scheduler_epoch_no_warmup(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=0.5, 
        cycle_mult=1, 
        warmup_period=0, 
        max_lr=[100, 200, 300], 
        min_lr=10, 
        gamma=1, 
        strategy='epoch'
    )
    lrs = []
    epoch = 2
    n_steps_per_epoch = 50

    for i in range(epoch):
        for j in range(n_steps_per_epoch):
            lrs.append(scheduler.get_last_lr())
            scheduler.step((i * n_steps_per_epoch + j + 1) / n_steps_per_epoch)
    
    assert lrs[0][0] == 100
    assert lrs[0][1] == 200
    assert lrs[0][2] == 300
    assert lrs[50][0] == 100
    assert lrs[50][1] == 200
    assert lrs[50][2] == 300

def test_scheduler_epoch_cycle_mult(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=0.5, 
        cycle_mult=2, 
        warmup_period=0, 
        max_lr=[100, 200, 300], 
        min_lr=10, 
        gamma=1, 
        strategy='epoch'
    )
    lrs = []
    epoch = 2
    n_steps_per_epoch = 50

    for i in range(epoch):
        for j in range(n_steps_per_epoch):
            lrs.append(scheduler.get_last_lr())
            scheduler.step((i * n_steps_per_epoch + j + 1) / n_steps_per_epoch)
    
    assert lrs[0][0] == 100
    assert lrs[0][1] == 200
    assert lrs[0][2] == 300
    assert lrs[75][0] == 100
    assert lrs[75][1] == 200
    assert lrs[75][2] == 300

def test_scheduler_epoch_gamma(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=0.5, 
        cycle_mult=1, 
        warmup_period=0, 
        max_lr=[100, 200, 300], 
        min_lr=10, 
        gamma=0.5, 
        strategy='epoch'
    )
    lrs = []
    epoch = 2
    n_steps_per_epoch = 50

    for i in range(epoch):
        for j in range(n_steps_per_epoch):
            lrs.append(scheduler.get_last_lr())
            scheduler.step((i * n_steps_per_epoch + j + 1) / n_steps_per_epoch)
    
    assert lrs[0][0] == 100
    assert lrs[0][1] == 200
    assert lrs[0][2] == 300
    assert lrs[50][0] == 25
    assert lrs[50][1] == 50
    assert lrs[50][2] == 75

def test_scheduler_epoch_warmup(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=0.5, 
        cycle_mult=1, 
        warmup_period=0.1, 
        max_lr=[100, 200, 300], 
        min_lr=10, 
        gamma=1, 
        strategy='epoch'
    )
    lrs = []
    epoch = 2
    n_steps_per_epoch = 50

    for i in range(epoch):
        for j in range(n_steps_per_epoch):
            lrs.append(scheduler.get_last_lr())
            scheduler.step((i * n_steps_per_epoch + j + 1) / n_steps_per_epoch)
    
    assert lrs[0][0] == 10
    assert lrs[0][1] == 10
    assert lrs[0][2] == 10
    assert lrs[25][0] == 10
    assert lrs[25][1] == 10
    assert lrs[25][2] == 10
    assert lrs[55][0] == 100
    assert lrs[55][1] == 200
    assert lrs[55][2] == 300

def test_scheduler_epoch_warmup_once(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=0.5, 
        cycle_mult=1, 
        warmup_period=0.1, 
        max_lr=[100, 200, 300], 
        min_lr=10, 
        warmup_once=True,
        gamma=1, 
        strategy='epoch'
    )
    lrs = []
    epoch = 2
    n_steps_per_epoch = 50

    for i in range(epoch):
        for j in range(n_steps_per_epoch):
            lrs.append(scheduler.get_last_lr())
            scheduler.step((i * n_steps_per_epoch + j + 1) / n_steps_per_epoch)
    
    assert lrs[0][0] == 10
    assert lrs[0][1] == 10
    assert lrs[0][2] == 10
    assert lrs[5][0] == 100
    assert lrs[5][1] == 200
    assert lrs[5][2] == 300
    assert lrs[75][0] == 100
    assert lrs[75][1] == 200
    assert lrs[75][2] == 300

def test_scheduler_epoch_combination(optimizer: MockOptimizer):
    scheduler = CosineAnnealingWithWarmRestarts(
        optimizer, 
        cycle_period=0.5, 
        cycle_mult=2, 
        warmup_period=0.1, 
        max_lr=[100, 200, 300], 
        min_lr=10, 
        gamma=0.5, 
        warmup_once=True,
        strategy='epoch'
    )
    lrs = []
    epoch = 2
    n_steps_per_epoch = 50

    for i in range(epoch):
        for j in range(n_steps_per_epoch):
            lrs.append(scheduler.get_last_lr())
            scheduler.step((i * n_steps_per_epoch + j + 1) / n_steps_per_epoch)
    
    assert lrs[0][0] == 10
    assert lrs[0][1] == 10
    assert lrs[0][2] == 10
    assert lrs[5][0] == 100
    assert lrs[5][1] == 200
    assert lrs[5][2] == 300
    assert lrs[75][0] == 25
    assert lrs[75][1] == 50
    assert lrs[75][2] == 75