# QuantumMedx — QFF 2026 QML Track

## Project overview

This project evaluates classical and quantum-kernel classification approaches
for the QFF high-dimensional Q2 classification task and defines a custom
Processor D for the official QFT-3 benchmark.

## Dataset

- Training file: `data/q2/hd_data_train.csv`
- Test features: `data/q2/hd_data_test_features.csv`
- Training set: 800 labeled examples with 50 features (`x01`–`x50`)
- Test set: 200 unlabeled examples
- `sample_id` is an identifier, not a model feature.
- Test labels must not be used for training or model selection.

## Q2 model comparison

Validation results recorded from the current training run:

| Model | Balanced accuracy | Macro F1 |
|---|---:|---:|
| RBF SVM, 50 features | 0.710687 | 0.710874 |
| RBF SVM, 6 features | 0.538570 | 0.496855 |
| Six-qubit quantum fidelity kernel | 0.566500 | 0.565884 |

The 50-feature RBF SVM was selected based on validation performance.
The quantum-kernel experiment uses a simulated quantum circuit and a
classical SVM; it is not a claim of execution on physical quantum hardware.

## Processor D

The configuration is in `processor_d_config.json`. Its QFT-3 benchmark
was executed with the official QFF benchmark code and the results were
consistent with the values stored in the configuration.

- Noise model: `QFF2026-v2`
- Depth before transpilation: 6
- Depth after transpilation: 22
- Two-qubit gate count: 12
- SWAP overhead: 1
- Ideal fidelity: 1.000000
- Noisy fidelity: 0.603585

## Environment setup

Use Python 3.11 and PowerShell from the project directory.

The existing project environment is `.venv-qff`. To use it:

```powershell
& ".\.venv-qff\Scripts\python.exe" -m pip install -r requirements.txt
```

If `requirements.txt` is not present, create it from your verified
environment rather than assuming this command will work:

```powershell
& ".\.venv-qff\Scripts\python.exe" -m pip freeze > requirements.txt
```

Run the Q2 training script:

```powershell
& ".\.venv-qff\Scripts\python.exe" ".\train_q2.py"
```

Run the official Processor D benchmark:

```powershell
& ".\.venv-qff\Scripts\python.exe" ".\official_kit\qff_qml_benchmark.py" ".\processor_d_config.json"
```

Launch the notebook with Jupyter installed in the environment:

```powershell
& ".\.venv-qff\Scripts\python.exe" -m pip install notebook
& ".\.venv-qff\Scripts\python.exe" -m notebook
```

## Output files

- `predictions_q2.csv` — predictions for the 200 Q2 test examples.
- `processor_d_config.json` — Processor D design and benchmark results.
- `results/q2_validation_results.json` — saved validation metrics, if present.
- `solution.ipynb` — experiment notes and benchmark verification.
- `results.md` — reported results and completion checklist.

## Reproducibility and submission

Record the environment with:

```powershell
& ".\.venv-qff\Scripts\python.exe" -m pip freeze > requirements.txt
```

Validate the prediction file using the official kit validator:

```powershell
$kit = "$env:USERPROFILE\Downloads\QFF2026_QML_Track (2)\QFF2026_QML_Track\QFF2026_QML_Challenge_Kit"
& ".\.venv-qff\Scripts\python.exe" "$kit\tools\validate_predictions.py" q2 ".\predictions_q2.csv"
```

Before submission, verify that the official Task A benchmark has been run,
all required files are present, and the final public repository URL is
submitted through the QFF portal. Do not publish private credentials or
sensitive information.
