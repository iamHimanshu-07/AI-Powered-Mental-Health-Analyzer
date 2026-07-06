# 🧠 MindPulse.AI — Mental Health Emotion Detector

[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)

> **AI-powered, multi-label mental-health emotion detection from free-form text.**

MindPulse.AI is an end-to-end NLP project that:

1. Loads the `DepressionEmo` Reddit dataset (`anger`, `sadness`, `emptiness`, `hopelessness`, `worthlessness`, `loneliness`, `suicide intent`, `brain dysfunction (forget)`).
2. Trains a **TF-IDF (1-2 grams, 10k features) + OneVsRest(RandomForest)** multi-label classifier.
3. Serves predictions through an interactive **Streamlit** dashboard with adjustable threshold, per-emotion probability chart, and downloadable report.

> ⚠️ **Disclaimer:** This project is for educational and awareness purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. If you or someone you know is in crisis, please contact a qualified mental-health professional or a crisis helpline in your country.

---

## ✨ Features

- **Multi-label classification** — a single text can trigger several emotion labels simultaneously.
- **Adjustable prediction threshold** in the sidebar — sweep from 0.05 to 0.95.
- **Per-class probability bar chart** with a colour-coded severity map.
- **High-risk warning banner** if `suicide intent` or `hopelessness` is detected.
- **Downloadable text report** of every prediction.
- **Live model metrics** (F1 micro/macro, precision, recall) shown in the sidebar.
- **Reproducible training script** (`train_model.py`) — re-train from scratch in one command.

---

## 🗂 Project structure

```
MindPulse.AI/
├── app/
│   ├── app.py                 # Streamlit front-end
│   └── models/                # Trained artefacts (created by train_model.py)
│       ├── mental_health_model.pkl
│       └── mlb.pkl
├── DepressionEmo/             # Raw dataset + research scripts
│   └── Dataset/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── Notebook/
│   ├── EDA.ipynb
│   └── Multi-label Classifier - Threshold Tuned.ipynb
├── models/
│   └── metrics.json           # Test-set metrics from train_model.py
├── train_model.py             # Reproducible training script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick start

### 1. Clone & enter

```bash
git clone https://github.com/iamHimanshu-07/MindPulse.AI.git
cd MindPulse.AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (one-time, ~1 min on CPU)

```bash
python train_model.py
```

This produces `app/models/mental_health_model.pkl`, `app/models/mlb.pkl`, and `models/metrics.json`.

> The model artefacts are intentionally **not** committed to the repo — they are large, binary, and easy to regenerate. This keeps the repo light and reproducible.

### 5. Launch the Streamlit app

```bash
streamlit run app/app.py
```

Then open <http://localhost:8501> in your browser.

---

## 📊 Model performance

Test set (20 % hold-out, `random_state=42`):

| Metric | Score |
| --- | --- |
| Hamming loss | 0.2442 |
| Subset accuracy | 0.1267 |
| F1 (micro) | 0.7584 |
| F1 (macro) | 0.7069 |
| Precision (micro) | 0.6944 |
| Recall (micro) | 0.8354 |

*Baseline only — see the **Future work** section for planned upgrades.*

---

## 🧪 Using the app

1. Type or paste any text into the **Your text** box.
2. Adjust the **Prediction threshold** in the sidebar (default 0.40).
   - Higher threshold → fewer, more confident labels.
   - Lower threshold → more labels, more false positives.
3. Click **🔍 Analyze**.
4. Read the **Predicted emotions**, the **probability bar chart**, and (if relevant) the **high-risk warning**.
5. Optionally **Download report (.txt)** for the full audit trail.

---

## 🔁 Re-training & experimentation

`train_model.py` is the single source of truth for the model. To try a different classifier, edit the `pipeline` definition (e.g. swap `RandomForestClassifier` for `LogisticRegression(solver='liblinear')`) and re-run. Remember:

- Only estimators with `predict_proba` work with the current Streamlit UI (`RandomForest`, `LogisticRegression`, `GradientBoosting`, …). `LinearSVC` does **not** support `predict_proba` and would need a UI tweak.

---

## 🛣 Future work

- Replace TF-IDF + RF with a fine-tuned transformer (e.g. `distilroberta-base`) for stronger F1.
- Add SHAP / LIME explanations so each prediction is interpretable.
- Persist user history (with consent) for longitudinal trend analysis.
- Containerise with Docker & deploy to Streamlit Community Cloud.
- Multi-lingual support beyond English.

---

## 🤝 Contributing

1. Fork the repo.
2. Create a feature branch: `git checkout -b feat/awesome`.
3. Commit your changes: `git commit -m "feat: ..."`.
4. Push: `git push origin feat/awesome`.
5. Open a Pull Request.

---

## 📄 License

[Apache-2.0](LICENSE)
