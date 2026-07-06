"""
MindPulse.AI — Mental Health Emotion Detector
Streamlit front-end for the multi-label classifier trained by train_model.py.
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# --------------------------------------------------------------------------- #
# Page config (must be the first Streamlit call)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="MindPulse.AI — Mental Health Emotion Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
MODEL_PATH = MODELS_DIR / "mental_health_model.pkl"
MLB_PATH = MODELS_DIR / "mlb.pkl"
METRICS_PATH = APP_DIR.parent / "models" / "metrics.json"

# Severity / colour mapping for the visualisation. Kept here (not in the
# trained artefact) so design tweaks don't force a re-train.
SEVERITY_COLORS = {
    "anger": "#FF7F50",
    "sadness": "#4F81BD",
    "emptiness": "#9E9E9E",
    "hopelessness": "#8B0000",
    "worthlessness": "#6A0DAD",
    "loneliness": "#87CEEB",
    "suicide intent": "#D62728",
    "brain dysfunction (forget)": "#2CA02C",
}
HIGH_RISK_LABELS = {"suicide intent", "hopelessness"}


# --------------------------------------------------------------------------- #
# Cached resource loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run `python train_model.py` from the repository root first."
        )
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner="Loading label binariser...")
def load_mlb():
    return joblib.load(MLB_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text())


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(metrics: dict | None) -> dict:
    with st.sidebar:
        st.markdown("## 🧠 MindPulse.AI")
        st.caption("AI-powered multi-label mental-health emotion detector")
        st.markdown("---")

        st.markdown("### ⚙️ Settings")
        threshold = st.slider(
            "Prediction threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.40,
            step=0.05,
            help="Probability above which a label is treated as predicted.",
        )
        show_probs_table = st.checkbox("Show full probability table", value=False)
        show_metrics = st.checkbox("Show model metrics", value=True)
        st.markdown("---")

        st.markdown("### ℹ️ About")
        st.write(
            "Predicts the presence of emotional signals (anger, sadness, "
            "emptiness, hopelessness, worthlessness, loneliness, suicide "
            "intent, brain dysfunction) from free-form text."
        )
        st.warning(
            "⚠️ For educational & awareness use only. Not a substitute for "
            "professional medical advice.",
            icon="⚠️",
        )

        if show_metrics and metrics:
            with st.expander("📊 Model metrics", expanded=True):
                m1, m2 = st.columns(2)
                m1.metric("F1 (micro)", f"{metrics['f1_micro']:.3f}")
                m2.metric("F1 (macro)", f"{metrics['f1_macro']:.3f}")
                m1.metric("Precision", f"{metrics['precision_micro']:.3f}")
                m2.metric("Recall", f"{metrics['recall_micro']:.3f}")
                st.caption(
                    f"Trained on {metrics['n_train']:,} samples · "
                    f"evaluated on {metrics['n_test']:,} · "
                    f"{len(metrics['labels'])} labels"
                )

    return {"threshold": threshold, "show_probs_table": show_probs_table}


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def predict(pipeline, mlb, text: str, threshold: float):
    proba = pipeline.predict_proba([text])[0]
    prediction = (proba >= threshold).astype(int)
    labels = mlb.inverse_transform(np.array([prediction]))[0]
    return proba, list(labels)


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def render_chart(emotions: np.ndarray, proba: np.ndarray, threshold: float):
    order = np.argsort(proba)
    emotions_sorted = emotions[order]
    proba_sorted = proba[order]
    colors = [SEVERITY_COLORS.get(label, "#BDBDBD") for label in emotions_sorted]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(emotions))))
    bars = ax.barh(emotions_sorted, proba_sorted, color=colors, edgecolor="white")

    ax.axvline(threshold, color="#444", linestyle="--", linewidth=1, label=f"threshold = {threshold:.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_title("Emotion Prediction Probabilities", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=False)

    for bar, p in zip(bars, proba_sorted):
        ax.text(
            min(p + 0.02, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{p:.2f}",
            va="center",
            fontsize=9,
            color="#222",
        )

    fig.tight_layout()
    return fig


def make_report(text: str, predicted: list[str], proba: np.ndarray, emotions: np.ndarray) -> str:
    buf = io.StringIO()
    buf.write("MindPulse.AI — Mental Health Emotion Report\n")
    buf.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
    buf.write("Input text:\n")
    buf.write(text.strip() + "\n\n")
    buf.write("Predicted emotions:\n")
    buf.write(", ".join(predicted) if predicted else "(none above threshold)\n")
    buf.write("\nEmotion probabilities:\n")
    for emo, p in sorted(zip(emotions, proba), key=lambda kv: -kv[1]):
        flag = "  ⚠ HIGH RISK" if emo in HIGH_RISK_LABELS else ""
        buf.write(f"  {emo:<28s} {p:.4f}{flag}\n")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    metrics = load_metrics()
    settings = render_sidebar(metrics)

    st.title("🧠 Mental Health Emotion Detector")
    st.write(
        "Type or paste any text — a journal entry, a message, a thought — "
        "and MindPulse.AI will surface the emotional signals it detects."
    )

    examples = st.session_state.pop("_examples", None)  # type: ignore[attr-defined]
    default_text = examples if examples is not None else (
        "I feel empty inside, like nothing matters anymore. I can't focus on anything "
        "and I'm always tired. Sometimes I wonder if anyone would even notice if I "
        "was gone."
    )
    text_input = st.text_area("Your text", value=default_text, height=180)

    c1, c2, _ = st.columns([1, 1, 6])
    analyse = c1.button("🔍 Analyze", type="primary", use_container_width=True)
    clear = c2.button("🧹 Clear", use_container_width=True)
    if clear:
        st.rerun()

    if not analyse:
        return

    if not text_input.strip():
        st.warning("Please enter some text before analysing.", icon="✍️")
        return

    try:
        pipeline = load_pipeline()
        mlb = load_mlb()
    except FileNotFoundError as e:
        st.error(str(e), icon="🚨")
        st.stop()

    threshold = settings["threshold"]
    with st.spinner("Analysing..."):
        proba, predicted = predict(pipeline, mlb, text_input, threshold)

    st.markdown("### 🔍 Predicted emotions")
    if predicted:
        cols = st.columns(min(len(predicted), 4))
        for col, label in zip(cols, predicted):
            color = SEVERITY_COLORS.get(label, "#BDBDBD")
            col.markdown(
                f"<div style='background:{color}22;border-left:4px solid {color};"
                f"padding:8px 12px;border-radius:6px'><b>{label}</b></div>",
                unsafe_allow_html=True,
            )
        if any(label in HIGH_RISK_LABELS for label in predicted):
            st.error(
                "⚠️ High-risk emotional indicators detected. "
                "If this is real, please consider reaching out to a professional "
                "or a crisis helpline in your country.",
                icon="🆘",
            )
    else:
        st.info(
            f"No strong emotions detected above the {threshold:.2f} threshold. "
            "Try lowering the threshold in the sidebar.",
            icon="ℹ️",
        )

    st.markdown("### 📊 Probability breakdown")
    fig = render_chart(mlb.classes_, proba, threshold)
    st.pyplot(fig)
    plt.close(fig)

    if settings["show_probs_table"]:
        import pandas as pd
        df = (
            pd.DataFrame({"emotion": mlb.classes_, "probability": proba})
            .sort_values("probability", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### 📥 Export")
    report = make_report(text_input, predicted, proba, mlb.classes_)
    st.download_button(
        label="📄 Download report (.txt)",
        data=report,
        file_name="mindpulse_report.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()
