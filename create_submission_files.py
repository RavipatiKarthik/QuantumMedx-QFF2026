from pathlib import Path
import json

root = Path(__file__).resolve().parent

readme = r"""# QuantumMedx — QFF 2026 QML Track

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
"""

results = r"""# QuantumMedx — QFF 2026 Results

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
"""

def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True)
    }

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True)
    }

cells = [
    md("""# QuantumMedx — QFF 2026 QML Track

This notebook documents the Q2 classification experiments and verifies the
official Task A and Task B benchmarks. Run the cells from the project root
with the `.venv-qff` environment.

**Important:** Task A results are calculated by the official benchmark code.
This notebook does not invent benchmark values or use hidden test labels.
"""),
    md("## 1. Check the project files"),
    code("""from pathlib import Path

root = Path.cwd()
required = [
    "train_q2.py",
    "predictions_q2.csv",
    "processor_d_config.json",
    "official_kit/qff_qml_benchmark.py",
    "data/q2/hd_data_train.csv",
    "data/q2/hd_data_test_features.csv",
]
for name in required:
    print(f"{name}: {(root / name).exists()}")
"""),
    md("## 2. Review Q2 validation results"),
    code("""import json
from pathlib import Path

metrics_path = Path("results/q2_validation_results.json")
if metrics_path.exists():
    print(json.dumps(json.loads(metrics_path.read_text(encoding="utf-8")),
                     indent=2))
else:
    print("Validation JSON not found. Run train_q2.py and check its output.")
"""),
    md("""## 3. Inspect prediction file

This cell checks the prediction file's row count and class distribution. It
does not measure test accuracy because the hidden test labels are unavailable.
"""),
    code("""import pandas as pd

predictions = pd.read_csv("predictions_q2.csv")
display(predictions.head())
print("Rows:", len(predictions))
print("Columns:", predictions.columns.tolist())
print("Prediction counts:")
display(predictions.iloc[:, -1].value_counts(dropna=False))
"""),
    md("""## 4. Task A — official Bell-state benchmark

The official kit computes ideal and noisy Bell-state fidelity estimates using
8192 shots per correlation. Execute this cell and preserve the printed values
for `results.md`. The official noise model is used without changing its settings.
"""),
    code("""import sys
from pathlib import Path

sys.path.insert(0, str(Path("official_kit").resolve()))
from qff_qml_benchmark import task_a_bell

task_a_results = task_a_bell(shots=8192)
print(json.dumps(task_a_results, indent=2))
"""),
    md("## 5. Task B — run the official Processor D benchmark"),
    code("""from qff_qml_benchmark import (
    load_processor_d, validate_processor_d, task_b_qft3
)

config = load_processor_d("processor_d_config.json")
design_errors = validate_processor_d(config, require_results=False)
print("Design validation errors:", design_errors)
if design_errors:
    raise ValueError("Processor D design validation failed.")
task_b_results = task_b_qft3(config)
print(json.dumps(task_b_results, indent=2))
"""),
    md("## 6. Final Processor D results validation"),
    code("""config = load_processor_d("processor_d_config.json")
validation_errors = validate_processor_d(config, require_results=True)
print("Validation errors:", validation_errors)
print("FINAL VALIDATION:", "PASS" if not validation_errors else "FAIL")
"""),
    md("""## 7. Interpretation and limitations

The recorded validation results favor the 50-feature RBF SVM over the
six-feature baseline and the six-qubit simulated quantum fidelity-kernel
experiment. Validation metrics are not hidden-test performance. The quantum
kernel is simulated; these results do not establish a quantum advantage.
Processor D metrics must be reported from the official benchmark output.
""")
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (QFF .venv-qff)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python", "version": "3.11"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

(root / "README.md").write_text(readme.strip() + "\n", encoding="utf-8")
(root / "results.md").write_text(results.strip() + "\n", encoding="utf-8")
(root / "solution.ipynb").write_text(
    json.dumps(notebook, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8"
)

print("Created README.md")
print("Created results.md")
print("Created solution.ipynb")
