
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = __import__("pathlib").Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "q2" / "hd_data_train.csv")

features = [f"x{i:02d}" for i in range(1, 51)]
X = df[features]
y = df["label"].astype(str)

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42,
)

models = {
    "Original RBF SVM": Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
        )),
    ]),
    "Tuned SVM": Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=30)),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma=0.01,
            class_weight="balanced",
        )),
    ]),
}

for name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": "f1_macro",
        },
        n_jobs=-1,
    )

    ba = scores["test_balanced_accuracy"]
    f1 = scores["test_macro_f1"]

    print(f"\n{name}")
    print(f"Balanced accuracy: {ba.mean():.4f} +/- {ba.std():.4f}")
    print(f"Macro F1:          {f1.mean():.4f} +/- {f1.std():.4f}")
