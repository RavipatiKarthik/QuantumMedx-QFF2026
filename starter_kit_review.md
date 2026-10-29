# QFF2026 QML Track — Starter Kit (kit v3)

Scaffold only — you build the actual classifier on top of this. Run from the `starter_kit/` folder inside the dedicated `qml-env` environment.

1. Environment check
2. Task A: Bell state — official fidelity from XX, YY, ZZ correlations
3. Task B: Processor D — design, validate, benchmark QFT-3
4. Running your own circuit under the benchmark noise model (Qiskit and PennyLane)
5. Blank QNode for your PS circuit
6. Writing the predictions file


```python
import sys, os
sys.path.insert(0, os.path.abspath('../noise_model'))
sys.path.insert(0, os.path.abspath('../benchmark'))
sys.path.insert(0, os.path.abspath('../tools'))
import pennylane as qml, qiskit, qiskit_aer, numpy as np, pandas as pd
print(f'Python:    {sys.version.split()[0]}')
print(f'PennyLane: {qml.__version__}')
print(f'Qiskit:    {qiskit.__version__}  Aer: {qiskit_aer.__version__}')
print(f'NumPy:     {np.__version__}   Pandas: {pd.__version__}')
SEED = 42
np.random.seed(SEED)
```

## Task A — Bell state |Φ+⟩

**Fidelity definition (official):** F = (1 + ⟨XX⟩ − ⟨YY⟩ + ⟨ZZ⟩) / 4. This is exact for |Φ+⟩ because |Φ+⟩⟨Φ+| = (II + XX − YY + ZZ)/4.

Do **not** use P(00)+P(11): that is only the ZZ population and ignores whether the superposition is coherent (a 50/50 classical mixture of |00⟩ and |11⟩ scores 1 on it but has F = 0.5).


```python
from qff_qml_benchmark import task_a_bell
task_a = task_a_bell()           # 8192 shots per basis, ideal and noisy
for cond in ('ideal', 'noisy'):
    r = task_a[cond]
    print(f"{cond:5s}: XX={r['XX']:+.4f}±{r['XX_se']:.4f}  YY={r['YY']:+.4f}±{r['YY_se']:.4f}  "
          f"ZZ={r['ZZ']:+.4f}±{r['ZZ_se']:.4f}  F={r['fidelity']:.4f}±{r['fidelity_se']:.4f}")
```

## Task B — Processor D

Edit the design below. Rules (enforced by the validator):

- 3–9 qubits; coupling map pairs are **directed** `[control, target]` and the direction is **enforced** by the benchmark. List both `[a,b]` and `[b,a]` for a two-way link.
- Native gates from: rx, ry, rz, h, x, y, z, s, sdg, t, tdg, sx, cx, cz (must include cx or cz).
- `benchmark_results` is required: it is filled in automatically below from the official QFT-3 benchmark.


```python
import json
from qff_qml_benchmark import validate_processor_d, task_b_qft3

proc_d = {
    'processor_id': 'YOUR_TEAM_proc_v1',                              # CHANGE
    'n_qubits': 5,                                                    # CHANGE (3-9)
    'coupling_map': [[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,4],[4,3]],  # CHANGE (directed pairs)
    'native_gate_set': ['rx','ry','rz','cx'],
    'qubit_layout': 'linear',
    'noise_assumptions': {'single_qubit_error_rate': 0.001, 'two_qubit_error_rate': 0.01,
                          'notes': 'Your own assumptions (do not affect the benchmark).'},
    'design_rationale': 'Explain in 3-6 sentences why this topology and gate set suit your PS circuit.',
}
problems = validate_processor_d(proc_d, require_results=False)
print('Design problems:', problems or 'none')
if not problems:
    proc_d['benchmark_results'] = task_b_qft3(proc_d)
    print(json.dumps(proc_d['benchmark_results'], indent=2))
    assert not validate_processor_d(proc_d)
    with open('processor_d_config.json', 'w') as f:
        json.dump(proc_d, f, indent=2)
    print('Saved processor_d_config.json')
```

## Running your own circuits under the benchmark noise model

The noise model only adds errors to specific gates. **Always** transpile to a fully-noised basis first, or gates such as `sx`, `u`, `cp`, `rzz` would run noise-free and your noisy results would look better than they are. The helpers below do this for you and raise an error if any un-noised gate remains.


```python
from qiskit import QuantumCircuit
from benchmark_noise_model import run_noisy, prepare_for_noisy_run, unnoised_gates, pennylane_noisy_device

qc = QuantumCircuit(2, 2)
qc.rzz(0.7, 0, 1); qc.sx(0)                     # gates the noise model does not cover directly
qc.measure([0, 1], [0, 1])
print('un-noised gates before transpiling:', unnoised_gates(qc))
print('after prepare_for_noisy_run      :', unnoised_gates(prepare_for_noisy_run(qc)))
print('noisy counts:', run_noisy(qc, shots=2048))

# PennyLane: same noise, same basis (needs the pinned PennyLane-qiskit plugin)
dev_noisy = pennylane_noisy_device(wires=2, shots=8192)
@qml.qnode(dev_noisy)
def zz_noisy():
    qml.Hadamard(0); qml.CNOT([0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
print('PennyLane noisy <ZZ>:', zz_noisy())
```


```python
# Blank QNode — replace with your actual circuit (ideal simulation)
n_qubits, n_layers = 4, 2
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev)
def ps_circuit(x, params):
    for i in range(n_qubits): qml.RY(x[i], wires=i)            # angle encoding
    for l in range(n_layers):
        for q in range(n_qubits): qml.RX(params[l,q,0], wires=q); qml.RY(params[l,q,1], wires=q)
        for q in range(n_qubits-1): qml.CNOT(wires=[q, q+1])
    return qml.expval(qml.PauliZ(0))

rng = np.random.default_rng(SEED)
out = ps_circuit(rng.normal(size=n_qubits), rng.normal(size=(n_layers, n_qubits, 2)))
print(f'Circuit output: {float(out):.4f}')
```

## Predictions file

- Test files start with a `sample_id` column (s0, s1, …). Drop it before feeding features to your model, and keep it for the output.
- Q1 / Q2: columns `sample_id,label`. Q3: `sample_id,label,score`, where `score` is your model's probability that label = 1 (ROC-AUC is computed from it).
- Check the file with `python tools/validate_predictions.py <q1|q2|q3> <file>`.


```python
from validate_predictions import validate
test = pd.read_csv('../data/q3/qnoise_test_features.csv')
X_test = test.drop(columns=['sample_id']).to_numpy()
# --- replace with your model: probability of label 1 for each test row ---
score = np.full(len(test), 0.5)
pred = pd.DataFrame({'sample_id': test['sample_id'], 'label': (score >= 0.5).astype(int), 'score': score})
pred.to_csv('predictions_q3_example.csv', index=False)
print('validator:', validate('q3', 'predictions_q3_example.csv') or 'OK')
```

## Next steps
1. Load data: `pd.read_csv('../data/q1/ecg_train.csv')`
2. Feature engineering / PCA (fit on train only)
3. Define loss and optimizer; train the circuit
4. Select models on the validation set only
5. Evaluate ideal and under the benchmark noise model (`pennylane_noisy_device` or `run_noisy`)
6. Write and validate the predictions file
7. Fill in `submission_templates/results_template.md`
