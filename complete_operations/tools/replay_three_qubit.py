"""Evaluate the supplied three-qubit control path on a finite gain grid."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import cnot_tolerance_v1 as model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', required=True, type=Path, help='new JSON output file')
args = parser.parse_args()
records = []
with np.load(ROOT / 'data/three_qubit_controls.npz', allow_pickle=False) as controls:
    for name in ('C_s_-0.04', 'C_s_-0.02', 'whole', 'C_s_+0.02', 'C_s_+0.04'):
        sequence = model.prepare(controls[name])
        nominal, responses = model.propagate(sequence, responses=True)
        errors = [model.halfdiamond(model.propagate(sequence, (x, y)))
                  for x in np.linspace(-.01, .01, 5)
                  for y in np.linspace(-.01, .01, 5)]
        records.append({
            'control': name,
            'nominal_half_diamond': model.halfdiamond(nominal),
            'first_order_generator_norms': [float(np.linalg.norm(k, 2)) for k in responses],
            'sampled_5x5_gain_grid_maximum': max(errors),
            'gate_time_t0': model.gate_time(controls[name]),
        })
with args.out.open('x', encoding='utf-8') as handle:
    json.dump({'precision': 'float64 / complex128',
               'scope': 'Finite coherent gain-grid replay of the five supplied controls.',
               'records': records}, handle, indent=2)
print('Wrote', len(records), 'three-qubit control records to', args.out)
