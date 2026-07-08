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
def _ensure_model_on_disk() -> None:
    """Download the trained artefacts on first run if they aren't on disk.

    On Streamlit Community Cloud the 268 MB ``mental_health_model.pkl`` is
    too large to commit, so ``app/models/fetch_model.py`` pulls both
    ``mental_health_model.pkl`` and ``mlb.pkl`` from ``MODEL_URL`` (env var
    or Streamlit secret). The fetch only runs when files are missing, so
    the local dev workflow is unchanged.
    """
    if (
        MODEL_PATH.exists()
        and MODEL_PATH.stat().st_size > 0
        and MLB_PATH.exists()
        and MLB_PATH.stat().st_size > 0
    ):
        return
    # Lazy import so the app can still boot even if the downloader is
    # broken in some environment — we surface a clear error in that case.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_mp_fetch_model", MODELS_DIR / "fetch_model.py"
    )
    if spec is None or spec.loader is None:
        raise FileNotFoundError(
            f"Could not locate the model downloader at {MODELS_DIR / 'fetch_model.py'}"
        )
    fetch_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch_model)  # type: ignore[union-attr]

    progress = st.progress(0.0, text="Fetching model on first run…")

    def _cb(done: int, total) -> None:
        if total and total > 0:
            progress.progress(min(done / total, 1.0))
        else:
            # Indeterminate: just bump to a stable "in-progress" value.
            progress.progress(0.5)

    try:
        fetch_model.ensure_model(progress_cb=_cb)
    except Exception as e:
        progress.empty()
        st.error(
            f"Could not download the trained model: {e}\n\n"
            "Set the `MODEL_URL` environment variable (or Streamlit secret "
            "`MODEL_URL`) to the base URL where both `mental_health_model.pkl` "
            "and `mlb.pkl` are hosted side-by-side.",
            icon="🚨",
        )
        st.stop()
    progress.empty()


@st.cache_resource(show_spinner="Loading primary model...")
def load_pipeline():
    _ensure_model_on_disk()
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
    """Read ``models/metrics.json`` and normalise to the nested schema.

    The current ``train_model.py`` writes a nested dict with ``primary`` and
    ``interpretable`` sub-dicts (one per trained model) plus top-level
    ``n_train``, ``n_test``, ``labels``, ``threshold``. Older training runs
    (and a stale ``metrics.json`` on disk) wrote a *flat* dict where the
    primary model's metrics sat at the top level. The sidebar uses the
    nested form, so we normalise here so a fresh deploy works even if the
    metrics file is from a previous trainer.
    """
    if not METRICS_PATH.exists():
        return None
    raw = json.loads(METRICS_PATH.read_text())

    # Already in the new nested format — pass through.
    if isinstance(raw, dict) and "primary" in raw and isinstance(raw["primary"], dict):
        return raw

    # Old flat format: {f1_micro, f1_macro, ...}. Lift the RF (primary) metrics
    # into the ``primary`` sub-dict the UI expects. ``interpretable`` stays
    # missing so the sidebar just skips the second-model block.
    flat_keys = (
        "hamming_loss", "accuracy", "precision_micro", "recall_micro",
        "f1_micro", "f1_macro",
    )
    primary = {k: float(raw[k]) for k in flat_keys if k in raw}
    if not primary:
        return None
    return {
        "primary_model": raw.get("primary_model", "OneVsRest(RandomForest)"),
        "interpretable_model": raw.get("interpretable_model", "OneVsRest(LogisticRegression)"),
        "primary": primary,
        "labels": raw.get("labels", []),
        "n_train": int(raw.get("n_train", 0)),
        "n_test": int(raw.get("n_test", 0)),
        "threshold": float(raw.get("threshold", 0.4)),
    }


# --------------------------------------------------------------------------- #
# Skeleton + animation CSS
# --------------------------------------------------------------------------- #
def _inject_css() -> None:
    """Inject the small CSS overlay used by the custom HTML we render.

    The app is light-theme only — colours come straight from ``BRAND`` and
    there's no toggle. The overlay is intentionally scoped to our own
    ``.mp-*`` classes so it doesn't fight Streamlit's stylesheet.
    """
    card_bg = "#FFFFFF"
    text = BRAND["text"]
    muted = BRAND["muted"]
    border = "#E0D8EC"
    surface_alt = BRAND["surface_alt"]

    css = f"""
    <style>
    /* ---------- custom emotion card ---------- */
    .mp-emotion-card {{
        background:{card_bg};
        color:{text};
        border:1px solid {border};
        border-left:4px solid var(--mp-color, {BRAND['primary']});
        border-radius:10px;
        padding:10px 14px;
        margin-bottom:6px;
        box-shadow: 0 1px 2px rgba(0,0,0,.04);
        animation: mpSlideIn .45s ease-out both;
    }}
    .mp-emotion-card .mp-label {{ font-weight:600; color:{text}; }}
    .mp-emotion-card .mp-prob  {{ color:{muted}; font-size:.85em; }}
    .mp-emotion-card .mp-tag   {{
        display:inline-block; margin-left:6px; padding:1px 8px; border-radius:10px;
        background:var(--mp-color, {BRAND['primary']}); color:#fff; font-size:.7em;
        letter-spacing:.04em; text-transform:uppercase;
    }}

    /* ---------- contributor block ---------- */
    .mp-contrib {{
        background:{surface_alt};
        color:{text};
        border:1px solid {border};
        border-radius:10px;
        padding:10px 12px;
    }}
    .mp-contrib-title {{ font-weight:600; margin-bottom:6px; color:{text}; }}
    .mp-contrib-label {{ color:{muted}; font-size:.85em; margin-bottom:2px; }}
    .mp-contrib-chip {{
        display:inline-block;
        margin:2px 4px;
        padding:2px 8px;
        border-radius:10px;
        font-size:.85em;
    }}
    .mp-contrib-chip .mp-contrib-score {{
        opacity:.85;
        color:#ffffff;
    }}

    /* ---------- animated emotion card entrance ---------- */
    @keyframes mpSlideIn {{
        from {{ opacity:0; transform: translateY(8px) scale(.97); }}
        to   {{ opacity:1; transform: translateY(0)   scale(1); }}
    }}

    /* ---------- loading skeleton ---------- */
    @keyframes mpShimmer {{
        0%   {{ background-position: -400px 0; }}
        100% {{ background-position: 400px 0; }}
    }}
    .mp-skel {{
        background: linear-gradient(90deg, {border} 0%, {surface_alt} 50%, {border} 100%);
        background-size: 800px 100%;
        animation: mpShimmer 1.4s infinite linear;
        border-radius:8px;
        height:18px;
        margin:6px 0;
    }}
    .mp-skel-row {{ display:flex; gap:10px; margin:10px 0; }}
    .mp-skel-row > div {{ flex:1; height:60px; }}

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
            key="threshold",
            help="Probability above which a label is treated as predicted.",
        )
        show_explanations = st.checkbox("Show top contributing words", value=True)
        show_probs_table = st.checkbox("Show full probability table", value=False)
        show_metrics = st.checkbox("Show model metrics", value=True)
        st.markdown("---")

        # ``metrics`` can be ``None`` if ``models/metrics.json`` is missing
        # (e.g. the file lives under a gitignored directory on Streamlit
        # Cloud). Use safe accessors so the sidebar still renders.
        primary = (metrics or {}).get("primary") or {}
        interp = (metrics or {}).get("interpretable") or {}

        if metrics and primary:
            with st.expander("📊 Model metrics", expanded=show_metrics):
                st.caption(f"**Primary** · {metrics.get('primary_model', 'RF')}")
                m1, m2 = st.columns(2)
                m1.metric("F1 (micro)",   f"{primary.get('f1_micro', 0.0):.3f}")
                m2.metric("F1 (macro)",   f"{primary.get('f1_macro', 0.0):.3f}")
                m1.metric("Precision",    f"{primary.get('precision_micro', 0.0):.3f}")
                m2.metric("Recall",       f"{primary.get('recall_micro', 0.0):.3f}")
                if interp:
                    st.caption(f"**Interpretable** · {metrics.get('interpretable_model', 'LogReg')}")
                    m3, m4 = st.columns(2)
                    m3.metric("F1 (micro)", f"{interp.get('f1_micro', 0.0):.3f}")
                    m4.metric("F1 (macro)", f"{interp.get('f1_macro', 0.0):.3f}")
                st.caption(
                    f"Trained on {metrics.get('n_train', 0):,} samples · "
                    f"evaluated on {metrics.get('n_test', 0):,} · "
                    f"{len(metrics.get('labels', []))} labels"
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
def render_chart(emotions: np.ndarray, proba: np.ndarray, threshold: float):
    order = np.argsort(proba)
    emotions_sorted = emotions[order]
    proba_sorted = proba[order]
    # Color by confidence; fall back to severity map.
    colors = [
        confidence_color(float(p)) if p >= 0.05
        else SEVERITY_COLORS.get(label, "#BDBDBD")
        for label, p in zip(emotions_sorted, proba_sorted)
    ]
    bg = "#FFFFFF"
    fg = BRAND["text"]
    grid_c = "#E0D8EC"

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


def render_contributors(contributors: dict[str, list[tuple[str, float]]], predicted: list[str]) -> str:
    if not contributors or not predicted:
        return ""
    rows = []
    for label in predicted:
        words = contributors.get(label, [])
        if not words:
            continue
        accent = SEVERITY_COLORS.get(label, BRAND["primary"])
        chips = " ".join(
            f"<span class='mp-contrib-chip' "
            f"style='background:{accent};color:#ffffff;"
            f"border:1px solid {accent};'>"
            f"{w} <span class='mp-contrib-score' "
            f"style='color:rgba(255,255,255,.85);'>+{c:.2f}</span></span>"
            for w, c in words
        )
        rows.append(
            f"<div style='margin:6px 0;'>"
            f"<div class='mp-contrib-label'>{label}</div>"
            f"<div>{chips}</div></div>"
        )
    if not rows:
        return ""
    return (
        "<div class='mp-contrib'>"
        "<div class='mp-contrib-title'>🧩 Top contributing words</div>"
        + "".join(rows)
        + "</div>"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    metrics = load_metrics()

    # ---------- session-state defaults ----------
    if "text_input_value" not in st.session_state:
        st.session_state.text_input_value = (
            "I feel empty inside, like nothing matters anymore. I can't focus on anything "
            "and I'm always tired. Sometimes I wonder if anyone would even notice if I "
            "was gone."
        )
    if "show_skeleton" not in st.session_state:
        st.session_state.show_skeleton = False

    # Inject the small CSS overlay used by the custom HTML blocks (emotion
    # cards, contributor chips) so they pick up the brand palette and stay
    # readable against Streamlit's default light background.
    _inject_css()

    settings = render_sidebar(metrics)

    st.title("🧠 Mental Health Emotion Detector")

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
    fig = render_chart(mlb.classes_, proba, threshold)
    st.pyplot(fig)
    plt.close(fig)

    # ---------- top contributing words (interpretable model) ----------
    if settings["show_explanations"]:
        st.markdown("### 🧩 Why these predictions?")
        st.caption(
            "Words from your text that pushed each label up, derived from "
            "a companion Logistic Regression model (TF-IDF × coefficient)."
        )
        contrib_html = render_contributors(contributors, predicted)
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
