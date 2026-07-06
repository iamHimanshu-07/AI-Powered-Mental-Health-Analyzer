"""
MindPulse.AI — Mental Health Emotion Detector
Streamlit front-end for the multi-label classifier trained by train_model.py.
"""

from __future__ import annotations

import io
import json
import re
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
INTERP_PATH = MODELS_DIR / "interpretable_model.pkl"
MLB_PATH = MODELS_DIR / "mlb.pkl"
METRICS_PATH = APP_DIR.parent / "models" / "metrics.json"

# --------------------------------------------------------------------------- #
# Brand palette & label metadata
# --------------------------------------------------------------------------- #
BRAND = {
    "primary": "#6A0DAD",       # deep purple
    "primary_light": "#9B6BD8",
    "surface": "#F7F5FB",
    "surface_alt": "#EDE6F5",
    "text": "#1E1B2E",
    "muted": "#6B6585",
    "danger": "#D62728",
}

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


def confidence_color(prob: float) -> str:
    """Green / yellow / red / gray for confidence-aware UI."""
    if prob >= 0.70:
        return "#2CA02C"  # green
    if prob >= 0.40:
        return "#E8A33D"  # amber
    if prob >= 0.20:
        return "#D62728"  # red
    return "#9E9E9E"      # gray


# --------------------------------------------------------------------------- #
# Curated examples (used by the "Try this" gallery)
# --------------------------------------------------------------------------- #
EXAMPLES = [
    {
        "label": "😔 Sad & hopeless",
        "text": "I feel so alone and empty, like nothing matters anymore. I "
                "can't focus on anything and I'm always tired. Sometimes I "
                "wonder if anyone would even notice if I was gone.",
        "tag": "high-risk",
    },
    {
        "label": "😡 Anger",
        "text": "I am absolutely furious that nobody bothered to listen to "
                "me. They just walked away like I was nothing.",
        "tag": "anger",
    },
    {
        "label": "🧠 Brain fog",
        "text": "My thoughts are loud and chaotic. I cannot focus or "
                "remember anything. I feel like I'm losing my mind.",
        "tag": "cognitive",
    },
    {
        "label": "😐 Neutral",
        "text": "Today was a nice walk in the park. The weather was fine "
                "and I had a sandwich for lunch.",
        "tag": "neutral",
    },
    {
        "label": "😞 Worthlessness",
        "text": "I'm such a failure. I can't do anything right. Everyone "
                "would be better off without me dragging them down.",
        "tag": "low-self",
    },
    {
        "label": "🌅 Hopeful",
        "text": "It was a hard week, but I went for a run this morning and "
                "felt a bit more like myself. Small steps.",
        "tag": "recovery",
    },
]


# --------------------------------------------------------------------------- #
# Cached resource loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading primary model...")
def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run `python train_model.py` from the repository root first."
        )
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner="Loading interpretable model...")
def load_interpretable():
    if not INTERP_PATH.exists():
        return None
    return joblib.load(INTERP_PATH)


@st.cache_resource(show_spinner="Loading label binariser...")
def load_mlb():
    return joblib.load(MLB_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text())


# --------------------------------------------------------------------------- #
# Word-cloud + theme CSS
# --------------------------------------------------------------------------- #
def _wordcloud_html(text: str, emotions_present: list[str], dark: bool) -> str:
    """Build a colour-coded HTML 'word cloud' that highlights words associated
    with the emotions predicted above the threshold. Falls back to a simple
    frequency list if wordcloud lib isn't available."""
    from collections import Counter
    import re

    stop = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "i",
        "you", "he", "she", "it", "we", "they", "to", "of", "in", "on", "for",
        "with", "at", "by", "from", "as", "this", "that", "these", "those",
        "be", "been", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "my", "your",
        "me", "him", "her", "us", "them", "so", "if", "not", "no", "yes",
        "am", "than", "then", "just", "about", "what", "how", "when", "where",
        "why", "who", "which", "all", "any", "some", "more", "most", "very",
        "too", "also", "into", "out", "up", "down", "over", "under", "i'm",
        "im", "dont", "don't", "cant", "can't", "its", "it's",
    }
    tokens = re.findall(r"[a-zA-Z']{3,}", text.lower())
    tokens = [t for t in tokens if t not in stop]
    counts = Counter(tokens).most_common(20)
    if not counts:
        return "<p style='opacity:.6'>Not enough words to render a cloud.</p>"

    bg = "#1E1B2E" if dark else "#F7F5FB"
    fg = "#F2EEFA" if dark else "#1E1B2E"
    max_count = counts[0][1]
    chips = []
    for word, c in counts:
        size = 0.9 + 2.4 * (c / max_count)
        # Tint the chip with the first predicted emotion's colour, otherwise brand.
        if emotions_present:
            base = SEVERITY_COLORS.get(emotions_present[0], BRAND["primary"])
        else:
            base = BRAND["primary_light"]
        chips.append(
            f"<span style='display:inline-block;margin:4px 6px;padding:4px 10px;"
            f"border-radius:14px;background:{base}22;color:{base};"
            f"border:1px solid {base}55;font-size:{size:.2f}em;'>"
            f"{word}<span style='opacity:.55;font-size:.7em;margin-left:6px'>×{c}</span></span>"
        )
    return (
        f"<div style='background:{bg};color:{fg};padding:14px;border-radius:10px;"
        f"border:1px solid {BRAND['surface_alt']};line-height:1.8;'>"
        + "".join(chips)
        + "</div>"
    )


# --------------------------------------------------------------------------- #
# Skeleton + animation CSS
# --------------------------------------------------------------------------- #
def _inject_css(dark: bool) -> None:
    bg = "#15131F" if dark else BRAND["surface"]
    card_bg = "#221C30" if dark else "#FFFFFF"
    text = "#F2EEFA" if dark else BRAND["text"]
    muted = "#A39DBA" if dark else BRAND["muted"]
    sidebar_bg = "#1B1726" if dark else BRAND["surface_alt"]
    border = "#2E2640" if dark else "#E0D8EC"

    css = f"""
    <style>
    /* ---------- page-level palette ---------- */
    .stApp {{ background:{bg}; color:{text}; }}
    section[data-testid="stSidebar"] > div {{ background:{sidebar_bg}; }}
    .stMarkdown, .stText, p, label, span {{ color:{text}; }}
    .stCaption, small {{ color:{muted} !important; }}
    hr {{ border-color:{border} !important; }}

    /* ---------- animated emotion card ---------- */
    @keyframes mpSlideIn {{
        from {{ opacity:0; transform: translateY(8px) scale(.97); }}
        to   {{ opacity:1; transform: translateY(0)   scale(1); }}
    }}
    .mp-emotion-card {{
        animation: mpSlideIn .45s ease-out both;
        background:{card_bg};
        border-radius:10px;
        padding:10px 14px;
        border-left:4px solid var(--mp-color, {BRAND['primary']});
        box-shadow: 0 1px 2px rgba(0,0,0,.04);
        margin-bottom:6px;
    }}
    .mp-emotion-card .mp-label {{ font-weight:600; color:{text}; }}
    .mp-emotion-card .mp-prob  {{ color:{muted}; font-size:.85em; }}
    .mp-emotion-card .mp-tag   {{
        display:inline-block; margin-left:6px; padding:1px 8px; border-radius:10px;
        background:var(--mp-color, {BRAND['primary']}); color:#fff; font-size:.7em;
        letter-spacing:.04em; text-transform:uppercase;
    }}

    /* ---------- loading skeleton ---------- */
    @keyframes mpShimmer {{
        0%   {{ background-position: -400px 0; }}
        100% {{ background-position: 400px 0; }}
    }}
    .mp-skel {{
        background: linear-gradient(90deg, {border} 0%, {sidebar_bg} 50%, {border} 100%);
        background-size: 800px 100%;
        animation: mpShimmer 1.4s infinite linear;
        border-radius:8px;
        height:18px;
        margin:6px 0;
    }}
    .mp-skel-row {{ display:flex; gap:10px; margin:10px 0; }}
    .mp-skel-row > div {{ flex:1; height:60px; }}

    /* ---------- primary button ---------- */
    .stButton > button[kind="primary"] {{
        background:{BRAND['primary']};
        color:#fff; border:none; font-weight:600;
    }}
    .stButton > button[kind="primary"]:hover {{ filter:brightness(1.08); }}

    /* ---------- metric cards ---------- */
    [data-testid="stMetric"] {{
        background:{card_bg};
        border:1px solid {border};
        border-radius:10px; padding:10px 12px;
    }}

    /* ---------- example gallery buttons ---------- */
    .mp-ex {{
        display:block; width:100%; text-align:left;
        background:{card_bg}; color:{text};
        border:1px solid {border}; border-radius:8px;
        padding:8px 10px; margin:4px 0; cursor:pointer;
        font-size:.9em;
    }}
    .mp-ex:hover {{ border-color:{BRAND['primary']}; color:{BRAND['primary']}; }}

    /* hide the default streamlit header gradient */
    header[data-testid="stHeader"] {{ background:transparent; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def _skeleton(n_cards: int = 3) -> str:
    rows = (
        "<div class='mp-skel-row'>"
        + "".join("<div class='mp-skel' style='flex:1'></div>" for _ in range(min(n_cards, 4)))
        + "</div>"
    )
    bars = "".join(
        f"<div class='mp-skel' style='width:{60 + (i*7)%35}%; height:14px'></div>"
        for i in range(6)
    )
    return f"<div class='mp-skel' style='height:24px;width:40%'></div>{rows}{bars}"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(metrics: dict | None, dark: bool) -> dict:
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
            key="threshold",
            help="Probability above which a label is treated as predicted.",
        )
        show_wordcloud = st.checkbox("Show color-coded word cloud", value=True)
        show_explanations = st.checkbox("Show top contributing words", value=True)
        show_probs_table = st.checkbox("Show full probability table", value=False)
        show_metrics = st.checkbox("Show model metrics", value=True)
        st.markdown("---")

        if metrics:
            with st.expander("📊 Model metrics", expanded=show_metrics):
                st.caption(f"**Primary** · {metrics.get('primary_model', 'RF')}")
                m1, m2 = st.columns(2)
                m1.metric("F1 (micro)", f"{metrics['primary']['f1_micro']:.3f}")
                m2.metric("F1 (macro)", f"{metrics['primary']['f1_macro']:.3f}")
                m1.metric("Precision", f"{metrics['primary']['precision_micro']:.3f}")
                m2.metric("Recall", f"{metrics['primary']['recall_micro']:.3f}")
                if "interpretable" in metrics:
                    st.caption(f"**Interpretable** · {metrics.get('interpretable_model', 'LogReg')}")
                    m3, m4 = st.columns(2)
                    m3.metric("F1 (micro)", f"{metrics['interpretable']['f1_micro']:.3f}")
                    m4.metric("F1 (macro)", f"{metrics['interpretable']['f1_macro']:.3f}")
                st.caption(
                    f"Trained on {metrics['n_train']:,} samples · "
                    f"evaluated on {metrics['n_test']:,} · "
                    f"{len(metrics['labels'])} labels"
                )

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

    return {
        "threshold": threshold,
        "show_wordcloud": show_wordcloud,
        "show_explanations": show_explanations,
        "show_probs_table": show_probs_table,
    }


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def predict(pipeline, mlb, text: str, threshold: float):
    proba = pipeline.predict_proba([text])[0]
    prediction = (proba >= threshold).astype(int)
    labels = mlb.inverse_transform(np.array([prediction]))[0]
    return proba, list(labels)


# --------------------------------------------------------------------------- #
# Top-contributing words (interpretable model)
# --------------------------------------------------------------------------- #
def top_contributors(interpretable, mlb, text: str, top_k: int = 5) -> dict[str, list[tuple[str, float]]]:
    """For each predicted label, return the words in `text` that contributed
    most positively to that label (highest coef * tfidf product)."""
    if interpretable is None:
        return {}
    try:
        vectorizer = interpretable.named_steps["tfidf"]
        clf = interpretable.named_steps["clf"]
    except AttributeError:
        return {}
    vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    nonzero = vec.nonzero()[1]
    if len(nonzero) == 0:
        return {}

    contributors: dict[str, list[tuple[str, float]]] = {}
    for class_idx, label in enumerate(mlb.classes_):
        try:
            coefs = clf.estimators_[class_idx].coef_[0]
        except (AttributeError, IndexError):
            continue
        contribs = []
        for j in nonzero:
            word = feature_names[j]
            tfidf_val = vec[0, j]
            if isinstance(tfidf_val, np.ndarray):
                tfidf_val = tfidf_val[0]
            c = coefs[j] * float(tfidf_val)
            if c > 0:
                contribs.append((word, float(c)))
        contribs.sort(key=lambda kv: -kv[1])
        if contribs:
            contributors[label] = contribs[:top_k]
    return contributors


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def render_chart(emotions: np.ndarray, proba: np.ndarray, threshold: float, dark: bool):
    order = np.argsort(proba)
    emotions_sorted = emotions[order]
    proba_sorted = proba[order]
    # Color by confidence; fall back to severity map.
    colors = [
        confidence_color(float(p)) if p >= 0.05
        else SEVERITY_COLORS.get(label, "#BDBDBD")
        for label, p in zip(emotions_sorted, proba_sorted)
    ]
    bg = "#15131F" if dark else "#FFFFFF"
    fg = "#F2EEFA" if dark else "#1E1B2E"
    grid_c = "#2E2640" if dark else "#E0D8EC"

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(emotions))))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    bars = ax.barh(emotions_sorted, proba_sorted, color=colors, edgecolor=bg)

    ax.axvline(threshold, color=BRAND["primary"], linestyle="--", linewidth=1, label=f"threshold = {threshold:.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted probability", color=fg)
    ax.set_title("Emotion Prediction Probabilities", fontsize=13, fontweight="bold", color=fg)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(grid_c)
    ax.grid(axis="x", linestyle="--", alpha=0.4, color=grid_c)
    ax.legend(loc="lower right", frameon=False, labelcolor=fg)

    for bar, p in zip(bars, proba_sorted):
        ax.text(
            min(p + 0.02, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{p:.2f}",
            va="center",
            fontsize=9,
            color=fg,
        )

    fig.tight_layout()
    return fig


def make_report(text: str, predicted: list[str], proba: np.ndarray, emotions: np.ndarray,
                contributors: dict[str, list[tuple[str, float]]]) -> str:
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
    if contributors:
        buf.write("\nTop contributing words (interpretable model):\n")
        for label, words in contributors.items():
            if label not in predicted:
                continue
            buf.write(f"  {label}:\n")
            for w, c in words:
                buf.write(f"    {w:<20s} +{c:.4f}\n")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Animated emotion cards
# --------------------------------------------------------------------------- #
def render_emotion_cards(predicted: list[str], proba: np.ndarray, mlb) -> str:
    label_to_prob = dict(zip(mlb.classes_, proba))
    if not predicted:
        return (
            f"<div class='mp-emotion-card' style='--mp-color:{BRAND['primary_light']}'>"
            f"<span class='mp-label'>No strong emotions above the threshold.</span></div>"
        )
    cards = []
    for label in predicted:
        p = float(label_to_prob.get(label, 0.0))
        color = confidence_color(p)
        risk_tag = (
            "<span class='mp-tag' style='background:#D62728'>high risk</span>"
            if label in HIGH_RISK_LABELS
            else ""
        )
        cards.append(
            f"<div class='mp-emotion-card' style='--mp-color:{color}'>"
            f"<span class='mp-label'>{label}</span>{risk_tag}"
            f"<div class='mp-prob'>{p:.1%} confidence</div></div>"
        )
    return "\n".join(cards)


def render_contributors(contributors: dict[str, list[tuple[str, float]]], predicted: list[str], dark: bool) -> str:
    if not contributors or not predicted:
        return ""
    bg = "#1B1726" if dark else "#FFFFFF"
    border = "#2E2640" if dark else "#E0D8EC"
    text = "#F2EEFA" if dark else "#1E1B2E"
    muted = "#A39DBA" if dark else "#6B6585"
    rows = []
    for label in predicted:
        words = contributors.get(label, [])
        if not words:
            continue
        chips = " ".join(
            f"<span style='display:inline-block;margin:2px 4px;padding:2px 8px;"
            f"border-radius:10px;background:{SEVERITY_COLORS.get(label, BRAND['primary'])}22;"
            f"color:{SEVERITY_COLORS.get(label, BRAND['primary'])};"
            f"border:1px solid {SEVERITY_COLORS.get(label, BRAND['primary'])}55;"
            f"font-size:.85em;'>{w} <span style='opacity:.6'>+{c:.2f}</span></span>"
            for w, c in words
        )
        rows.append(
            f"<div style='margin:6px 0;'>"
            f"<div style='color:{muted};font-size:.85em;margin-bottom:2px'>{label}</div>"
            f"<div>{chips}</div></div>"
        )
    if not rows:
        return ""
    return (
        f"<div style='background:{bg};border:1px solid {border};border-radius:10px;"
        f"padding:10px 12px;color:{text};'>"
        f"<div style='font-weight:600;margin-bottom:6px'>🧩 Top contributing words</div>"
        + "".join(rows)
        + "</div>"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    metrics = load_metrics()

    # ---------- session-state defaults ----------
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    if "text_input_value" not in st.session_state:
        st.session_state.text_input_value = (
            "I feel empty inside, like nothing matters anymore. I can't focus on anything "
            "and I'm always tired. Sometimes I wonder if anyone would even notice if I "
            "was gone."
        )
    if "show_skeleton" not in st.session_state:
        st.session_state.show_skeleton = False

    settings = render_sidebar(metrics, st.session_state.dark_mode)
    _inject_css(st.session_state.dark_mode)

    # ---------- top bar: title + dark-mode toggle ----------
    title_col, toggle_col = st.columns([0.85, 0.15])
    with title_col:
        st.title("🧠 Mental Health Emotion Detector")
    with toggle_col:
        st.write("")  # vertical alignment nudge
        label = "☀️ Light" if st.session_state.dark_mode else "🌙 Dark"
        if st.button(label, use_container_width=True, key="dark_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.write(
        "Type or paste any text — a journal entry, a message, a thought — "
        "and MindPulse.AI will surface the emotional signals it detects."
    )

    # ---------- example gallery ----------
    st.markdown("##### ✨ Try a curated example")
    ex_cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        with ex_cols[i % 3]:
            if st.button(ex["label"], key=f"ex_{i}", use_container_width=True):
                st.session_state.text_input_value = ex["text"]
                st.rerun()

    # ---------- input ----------
    text_input = st.text_area(
        "Your text",
        value=st.session_state.text_input_value,
        height=180,
        key="text_input_value",
    )

    c1, c2, _ = st.columns([1, 1, 6])
    analyse = c1.button("🔍 Analyze", type="primary", use_container_width=True)
    clear = c2.button("🧹 Clear", use_container_width=True)
    if clear:
        st.session_state.text_input_value = ""
        st.rerun()

    if not analyse:
        return

    if not text_input.strip():
        st.warning("Please enter some text before analysing.", icon="✍️")
        return

    try:
        pipeline = load_pipeline()
        mlb = load_mlb()
        interpretable = load_interpretable()
    except FileNotFoundError as e:
        st.error(str(e), icon="🚨")
        st.stop()

    threshold = settings["threshold"]

    # ---------- loading skeleton (replaces the spinner briefly) ----------
    placeholder = st.empty()
    placeholder.markdown(
        f"<div>{_skeleton(n_cards=len(['x']*3))}</div>",
        unsafe_allow_html=True,
    )
    with st.spinner("Analysing..."):
        import time
        time.sleep(0.4)  # give the skeleton a moment to be visible
        proba, predicted = predict(pipeline, mlb, text_input, threshold)
        contributors = top_contributors(interpretable, mlb, text_input)
    placeholder.empty()

    # ---------- predicted emotion cards ----------
    st.markdown("### 🔍 Predicted emotions")
    st.markdown(render_emotion_cards(predicted, proba, mlb), unsafe_allow_html=True)
    if any(label in HIGH_RISK_LABELS for label in predicted):
        st.error(
            "⚠️ High-risk emotional indicators detected. "
            "If this is real, please consider reaching out to a professional "
            "or a crisis helpline in your country.",
            icon="🆘",
        )

    # ---------- probability chart ----------
    st.markdown("### 📊 Probability breakdown")
    fig = render_chart(mlb.classes_, proba, threshold, st.session_state.dark_mode)
    st.pyplot(fig)
    plt.close(fig)

    # ---------- color-coded word cloud ----------
    if settings["show_wordcloud"]:
        st.markdown("### ☁️ Color-coded word cloud")
        st.caption("Word chips sized by frequency, tinted with the top predicted emotion.")
        st.markdown(
            _wordcloud_html(text_input, predicted, st.session_state.dark_mode),
            unsafe_allow_html=True,
        )

    # ---------- top contributing words (interpretable model) ----------
    if settings["show_explanations"]:
        st.markdown("### 🧩 Why these predictions?")
        st.caption(
            "Words from your text that pushed each label up, derived from "
            "a companion Logistic Regression model (TF-IDF × coefficient)."
        )
        contrib_html = render_contributors(contributors, predicted, st.session_state.dark_mode)
        if contrib_html:
            st.markdown(contrib_html, unsafe_allow_html=True)
        else:
            st.info("No interpretable contributions available (model not loaded or text too short).")

    # ---------- probability table ----------
    if settings["show_probs_table"]:
        import pandas as pd
        df = (
            pd.DataFrame({"emotion": mlb.classes_, "probability": proba})
            .sort_values("probability", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ---------- export ----------
    st.markdown("### 📥 Export")
    report = make_report(text_input, predicted, proba, mlb.classes_, contributors)
    st.download_button(
        label="📄 Download report (.txt)",
        data=report,
        file_name="mindpulse_report.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()
