import json
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "q2"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

train_df = pd.read_csv(DATA / "hd_data_train.csv")
test_df = pd.read_csv(DATA / "hd_data_test_features.csv")

ALL_FEATURES = [f"x{i:02d}" for i in range(1, 51)]
Q_FEATURES = [f"x{i:02d}" for i in range(1, 7)]

X = train_df[ALL_FEATURES].to_numpy(dtype=float)
y = train_df["label"].astype(str).to_numpy()
X_test = test_df[ALL_FEATURES].to_numpy(dtype=float)

assert set(np.unique(y)) == {"A", "B"}
assert list(test_df.columns)[0] == "sample_id"

indices = np.arange(len(y))
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

Xtr, Xval = X[train_idx], X[val_idx]
ytr, yval = y[train_idx], y[val_idx]

# Quantum feature map: six qubits, angle encoding, ring entanglement.
N_QUBITS = 6
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def quantum_state(angles):
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)
        qml.RY(angles[i], wires=i)

    for i in range(N_QUBITS):
        j = (i + 1) % N_QUBITS
        qml.CZ(wires=[i, j])
        qml.RZ(angles[i] * angles[j], wires=j)

    for i in range(N_QUBITS):
        j = (i + 1) % N_QUBITS
        qml.CZ(wires=[i, j])

    return qml.state()

def scale_quantum_features(scaler, frame):
    values = scaler.transform(frame[Q_FEATURES].to_numpy(dtype=float))
    # Bound encoded angles and pairwise phase features.
    return np.pi * np.tanh(values / 3.0)

def state_vectors(angles):
    return np.asarray([np.asarray(quantum_state(row)) for row in angles])

def fidelity_kernel(states_a, states_b):
    overlaps = states_a.conj() @ states_b.T
    return np.abs(overlaps) ** 2

def evaluate(name, y_true, y_pred):
    return {
        "model": name,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro"))
    }

results = []

# Classical RBF-SVM baselines: all 50 features and the six signal features.
classical_models = {}

for feature_name, feature_list in [
    ("rbf_svm_50_features", ALL_FEATURES),
    ("rbf_svm_6_features", Q_FEATURES),
]:
    best = None

    for C in [0.1, 1.0, 10.0]:
        for gamma in ["scale", 0.1, 1.0]:
            model = Pipeline([
                ("scale", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=C, gamma=gamma))
            ])
            model.fit(
                train_df.iloc[train_idx][feature_list],
                ytr
            )
            pred = model.predict(train_df.iloc[val_idx][feature_list])
            score = balanced_accuracy_score(yval, pred)

            if best is None or score > best["score"]:
                best = {
                    "score": float(score),
                    "model": model,
                    "C": C,
                    "gamma": gamma,
                    "pred": pred
                }

    classical_models[feature_name] = best
    results.append(evaluate(feature_name, yval, best["pred"]))
    print(
        f"{feature_name}: validation balanced accuracy="
        f"{best['score']:.4f}; C={best['C']}; gamma={best['gamma']}"
    )

# Quantum-kernel SVM. Fit preprocessing on validation-training rows only.
q_scaler = StandardScaler()
q_scaler.fit(train_df.iloc[train_idx][Q_FEATURES])

q_train_angles = scale_quantum_features(
    q_scaler, train_df.iloc[train_idx]
)
q_val_angles = scale_quantum_features(
    q_scaler, train_df.iloc[val_idx]
)

print("Generating quantum state vectors for validation...")
q_train_states = state_vectors(q_train_angles)
q_val_states = state_vectors(q_val_angles)

K_train = fidelity_kernel(q_train_states, q_train_states)
K_val = fidelity_kernel(q_val_states, q_train_states)

best_q = None
for C in [0.1, 1.0, 10.0]:
    model = SVC(kernel="precomputed", C=C)
    model.fit(K_train, ytr)
    pred = model.predict(K_val)
    score = balanced_accuracy_score(yval, pred)

    if best_q is None or score > best_q["score"]:
        best_q = {
            "score": float(score),
            "C": C,
            "pred": pred,
            "model": model
        }

results.append(evaluate("quantum_kernel_6_qubits", yval, best_q["pred"]))
print(
    "quantum_kernel_6_qubits: validation balanced accuracy="
    f"{best_q['score']:.4f}; C={best_q['C']}"
)

# Select the model with the highest validation balanced accuracy.
# In a tie, preserve the order in results.
winner = max(results, key=lambda row: row["balanced_accuracy"])
print("\nValidation comparison:")
for row in results:
    print(
        f"  {row['model']}: balanced_accuracy="
        f"{row['balanced_accuracy']:.4f}, "
        f"macro_F1={row['f1_macro']:.4f}"
    )
print("Selected model:", winner["model"])

# Refit the selected model on all 800 labeled training rows.
if winner["model"] == "quantum_kernel_6_qubits":
    final_scaler = StandardScaler()
    final_scaler.fit(train_df[Q_FEATURES])

    all_angles = scale_quantum_features(final_scaler, train_df)
    test_angles = scale_quantum_features(final_scaler, test_df)

    print("Generating final quantum state vectors...")
    all_states = state_vectors(all_angles)
    test_states = state_vectors(test_angles)

    K_all = fidelity_kernel(all_states, all_states)
    K_test = fidelity_kernel(test_states, all_states)

    final_model = SVC(kernel="precomputed", C=best_q["C"])
    final_model.fit(K_all, y)
    test_predictions = final_model.predict(K_test)
    kernel_info = {
        "qubits": N_QUBITS,
        "encoded_features": Q_FEATURES,
        "training_samples": len(y),
        "test_samples": len(test_df),
        "training_kernel_pairs": int(len(y) * len(y)),
        "test_kernel_pairs": int(len(test_df) * len(y)),
        "kernel": "squared state-vector overlap",
        "simulation": "ideal statevector"
    }
else:
    feature_list = (
        ALL_FEATURES
        if winner["model"] == "rbf_svm_50_features"
        else Q_FEATURES
    )
    chosen = classical_models[winner["model"]]
    final_model = chosen["model"]
    final_model.fit(train_df[feature_list], y)
    test_predictions = final_model.predict(test_df[feature_list])
    kernel_info = None

predictions = pd.DataFrame({
    "sample_id": test_df["sample_id"],
    "label": test_predictions
})
pred_path = ROOT / "predictions_q2.csv"
predictions.to_csv(pred_path, index=False)

summary = {
    "seed": SEED,
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "validation_rows": int(len(val_idx)),
    "validation_results": results,
    "selected_model": winner["model"],
    "quantum_kernel_details": kernel_info,
    "test_labels_used_for_training_or_selection": False
}
(RESULTS / "q2_validation_results.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print(f"\nSaved predictions: {pred_path}")
print(f"Saved validation results: {RESULTS / 'q2_validation_results.json'}")
print("Prediction counts:")
print(predictions["label"].value_counts().to_string())
print("Prediction rows:", len(predictions))
