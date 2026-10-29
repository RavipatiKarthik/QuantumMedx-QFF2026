# QuantumMedx — QFF 2026 Results

## Q2 classification

The following are the recorded validation results from the current training
run. They are validation metrics, not a measurement of accuracy on hidden
test labels.

| Model | Balanced accuracy | Macro F1 |
|---|---:|---:|
| RBF SVM, 50 features | 0.710687 | 0.710874 |
| RBF SVM, 6 features | 0.538570 | 0.496855 |
| Six-qubit quantum fidelity kernel | 0.566500 | 0.565884 |

### Model selection

The RBF SVM using all 50 features was selected because it achieved the
highest recorded validation balanced accuracy and macro F1 among these
three experiments. The six-feature baseline performed worse, consistent
with useful information being distributed beyond the first six features.

The quantum-kernel experiment achieved lower validation scores than the
50-feature classical baseline. It is included as a comparative experiment,
not as a claim that the quantum model outperformed classical machine learning.

The quantum kernel was simulated and used with a classical SVM. No claim of
physical quantum-hardware execution is made.

## Q2 predictions

- Expected test examples: 200
- Prediction file: `predictions_q2.csv`
- The official format validator previously returned `OK - predictions file is valid.`

That validator confirms file-format compliance; it does not establish
predictive accuracy on the hidden test labels.

## Processor D — official Task B benchmark

The official QFT-3 benchmark was executed using
`official_kit/qff_qml_benchmark.py`. Its printed values matched the
configuration's recorded benchmark values.

| Metric | Measured value |
|---|---:|
| Circuit depth before transpilation | 6 |
| Circuit depth after transpilation | 22 |
| Two-qubit gate count | 12 |
| SWAP overhead | 1 |
| Ideal fidelity | 1.000000 |
| Noisy fidelity | 0.603585 |
| Noise model version | QFF2026-v2 |

The official Processor D validator returned no validation errors:
`VALIDATION ERRORS: []`, `FINAL VALIDATION: PASS`.

## Task A — Bell-state fidelity

**Pending execution and recording.** Run the Task A cell in `solution.ipynb`
to obtain the official ideal and noisy Bell-state fidelity estimates using
8192 shots per correlation. Copy the actual output here after execution.
Do not substitute illustrative or guessed values.

## Final checks

- [x] Q2 predictions generated.
- [x] Prediction file passed the official format validator.
- [x] Processor D official QFT-3 benchmark executed.
- [x] Processor D configuration passed official validation.
- [ ] Task A Bell-state benchmark executed and results recorded.
- [ ] Environment exported to `requirements.txt`.
- [ ] Final notebook executed and saved.
- [ ] Public repository and portal submission verified.

## Task A — Official Bell-State Benchmark

Benchmark noise model: QFF2026-v2
Shots per correlation: 8192

| Metric | Ideal | Noisy |
|---|---:|---:|
| XX correlation | 1.000000 | 0.819092 |
| YY correlation | -1.000000 | -0.827881 |
| ZZ correlation | 1.000000 | 0.873535 |
| Bell-state fidelity | 1.000000 | 0.880127 |
| Fidelity standard error | 0.000000 | 0.002592 |

The official benchmark reports an ideal fidelity of 1.0 and a noisy fidelity of approximately 0.8801 under QFF2026-v2. These are benchmark results, not evidence of a quantum advantage.

