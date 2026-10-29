
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "q2"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Load only the labeled training data.
df = pd.read_csv(DATA / "hd_data_train.csv")
features = [f"x{i:02d}" for i in range(1, 51)]

X = df[features]
y = df["label"].astype(str)

# Keep the same 80/20 split as the previous experiment.
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)

# Establish a reproducible baseline using the original search grid.
baseline_best = None

for C in [0.1, 1.0, 10.0]:
    for gamma in ["scale", 0.1, 1.0]:
        model = Pipeline([
            ("scale", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=C, gamma=gamma)),
        ])
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = balanced_accuracy_score(y_val, pred)

        if baseline_best is None or score > baseline_best["score"]:
            baseline_best = {
                "score": float(score),
                "C": C,
                "gamma": gamma,
                "model": model,
                "pred": pred,
            }

print("BASELINE — original search grid")
print("C:", baseline_best["C"])
print("gamma:", baseline_best["gamma"])
print("Balanced accuracy:", round(baseline_best["score"], 4))
print(
    "Macro F1:",
    round(f1_score(y_val, baseline_best["pred"], average="macro"), 4),
)

# All scaling and feature selection happen inside each CV fold.
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif)),
    ("svm", SVC(kernel="rbf")),
])

param_grid = {
    "select__k": [10, 20, 30, 40, "all"],
    "svm__C": [0.1, 1.0, 10.0, 100.0],
    "svm__gamma": ["scale", 0.001, 0.01, 0.1],
    "svm__class_weight": [None, "balanced"],
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring={
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
    },
    refit="balanced_accuracy",
    cv=cv,
    n_jobs=-1,
    return_train_score=False,
)

print("\nRunning 5-fold cross-validated hyperparameter search...")
search.fit(X_train, y_train)

# Evaluate the selected configuration on the held-out split.
tuned_pred = search.predict(X_val)
tuned_balanced_accuracy = balanced_accuracy_score(y_val, tuned_pred)
tuned_macro_f1 = f1_score(y_val, tuned_pred, average="macro")

print("\nTUNED MODEL")
print("Best parameters:", search.best_params_)
print("Mean CV balanced accuracy:", round(search.best_score_, 4))
print("Holdout balanced accuracy:", round(tuned_balanced_accuracy, 4))
print("Holdout macro F1:", round(tuned_macro_f1, 4))

summary = {
    "seed": SEED,
    "training_rows_for_search": len(X_train),
    "holdout_rows": len(X_val),
    "baseline": {
        "C": baseline_best["C"],
        "gamma": baseline_best["gamma"],
        "holdout_balanced_accuracy": float(baseline_best["score"]),
        "holdout_macro_f1": float(
            f1_score(y_val, baseline_best["pred"], average="macro")
        ),
    },
    "tuned_model": {
        "best_params": {
            key: str(value) if value == "all" else value
            for key, value in search.best_params_.items()
        },
        "mean_cv_balanced_accuracy": float(search.best_score_),
        "holdout_balanced_accuracy": float(tuned_balanced_accuracy),
        "holdout_macro_f1": float(tuned_macro_f1),
    },
    "test_labels_used": False,
    "predictions_q2_modified": False,
}

output_path = RESULTS / "q2_tuning_results.json"
output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\nSaved tuning report:", output_path)
print("Original training script and predictions were not modified.")
