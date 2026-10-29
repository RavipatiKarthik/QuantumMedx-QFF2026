import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "results" / "q2_validation_results.json"
PREDICTIONS_FILE = ROOT / "predictions_q2.csv"

st.set_page_config(
    page_title="QuantumMedx | QML Dashboard",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ QuantumMedx")
st.subheader("High-Dimensional Quantum-Kernel Classification")
st.caption("QFF 2026 · QML Track · Problem Q2")

if not RESULTS_FILE.exists() or not PREDICTIONS_FILE.exists():
    st.error("Required results files were not found. Run train_q2.py first.")
    st.stop()

with open(RESULTS_FILE, encoding="utf-8") as f:
    results = json.load(f)

predictions = pd.read_csv(PREDICTIONS_FILE)
comparison = pd.DataFrame(results["validation_results"])

best = max(
    results["validation_results"],
    key=lambda item: item["balanced_accuracy"]
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Training samples", results["train_rows"])
m2.metric("Test predictions", len(predictions))
m3.metric("Best validation score", f"{best['balanced_accuracy']:.2%}")
m4.metric("Selected model", "Classical RBF-SVM" if "rbf_svm" in results["selected_model"] else "Quantum Kernel")

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("📊 Model Performance")
    chart_data = comparison.set_index("model")[["balanced_accuracy"]].rename(
        columns={"balanced_accuracy": "Balanced accuracy"}
    )
    st.bar_chart(chart_data, y="Balanced accuracy", height=300)
    st.dataframe(
        comparison.rename(columns={
            "model": "Model",
            "balanced_accuracy": "Balanced accuracy",
            "f1_macro": "Macro F1"
        }),
        use_container_width=True,
        hide_index=True
    )

with right:
    st.subheader("🧠 Selected Model")
    st.write(results["selected_model"])
    st.write("Validation split:", results["validation_rows"], "samples")
    st.write("Random seed:", results["seed"])
    st.write("Test labels used for training:", "No")
    st.subheader("🏷️ Predicted Classes")
    counts = predictions["label"].value_counts().rename_axis("Label").to_frame("Count")
    st.bar_chart(counts)
    st.dataframe(counts, use_container_width=True)

st.divider()

st.subheader("🔎 Prediction Explorer")
st.caption("These are model predictions for the 200 unlabeled test samples—not verified test accuracy.")

search = st.text_input("Search by sample ID", placeholder="e.g. s10")
view = predictions.copy()

if search.strip():
    view = view[view["sample_id"].astype(str).str.contains(search.strip(), case=False)]

st.dataframe(view, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Download predictions_q2.csv",
    data=PREDICTIONS_FILE.read_bytes(),
    file_name="predictions_q2.csv",
    mime="text/csv"
)

st.divider()
st.caption(
    "QuantumMedx · Research prototype. Validation results are measured on a "
    "held-out split of the labeled training data. Official hidden-test performance "
    "is not available here."
)
