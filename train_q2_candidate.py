
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "data" / "q2" / "hd_data_train.csv"
TEST_PATH = ROOT / "data" / "q2" / "hd_data_test_features.csv"
OUTPUT_PATH = ROOT / "predictions_q2_candidate.csv"
REPORT_PATH = ROOT / "results" / "q2_candidate_results.json"

ID_COLUMN = "sample_id"
TARGET_COLUMN = "label"


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    if ID_COLUMN not in test_df.columns:
        raise ValueError(f"Missing test ID column: {ID_COLUMN}")

    feature_columns = [
        column for column in train_df.columns
        if column != TARGET_COLUMN
    ]

    missing_features = [
        column for column in feature_columns
        if column not in test_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Test data is missing features: {missing_features}"
        )

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[feature_columns]

    if len(train_df) != 800 or len(test_df) != 200:
        raise ValueError(
            f"Unexpected data sizes: train={len(train_df)}, "
            f"test={len(test_df)}"
        )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=30)),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma=0.01,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Preserve the official sample IDs and expected output columns.
    candidate = pd.DataFrame({
        ID_COLUMN: test_df[ID_COLUMN],
        TARGET_COLUMN: predictions
    })

    if candidate[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate sample IDs found in candidate output.")

    if candidate[TARGET_COLUMN].isna().any():
        raise ValueError("Candidate output contains missing predictions.")

    candidate.to_csv(OUTPUT_PATH, index=False)

    report = {
        "model": "tuned_rbf_svm",
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "selected_features": 30,
        "C": 1.0,
        "gamma": 0.01,
        "class_weight": "balanced",
        "test_labels_used": False,
        "original_predictions_modified": False,
        "prediction_counts": {
            str(label): int(count)
            for label, count in candidate[TARGET_COLUMN]
            .value_counts().items()
        },
        "candidate_file": OUTPUT_PATH.name
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("Candidate model training completed.")
    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Features before selection: {len(feature_columns)}")
    print(f"Predictions: {report['prediction_counts']}")
    print(f"Candidate saved to: {OUTPUT_PATH}")
    print(f"Report saved to: {REPORT_PATH}")
    print("Original predictions_q2.csv was not modified.")


if __name__ == "__main__":
    main()
