import sys
sys.path.insert(0, '..')
import dicty_engine as eng  # noqa: F401  (registers sense_adapt)
from plexus.models.registry import get_operator
from plexus.specs.loader import load_spec

# 1. operator is registered
op_cls = get_operator('sense_adapt')
print('sense_adapt registered:', op_cls.__name__)

# 2. spec loads + schedules sense_adapt
sc = load_spec('specs/dicty_loop_base.yaml')
print('spec ops:', [o.op for o in sc.operators])
print('schedule:', sc.schedule)

# 3. construct an instance with parent params (adapt_rate=0) and check ablation
import torch
inst = op_cls({'from': 'camp', 'gain': 40.0, 'adapt_rate': 0.0, 'adapt_recover': 0.02, 'adapt_thr': 0.05, 'dt': 0.5})
print('ablation built:', inst.adapt_rate, inst.gain)

# 4. run a few frames
H, hist = eng.run(sc, device='cpu')
print('frames:', len(hist), 'last n_active:', hist[-1]['count'])
