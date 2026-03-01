# 🎮 ChurnIQ — Intelligent Player Churn Prediction

> A Machine Learning-powered web dashboard that predicts which online game players are at risk of quitting — built with **Python**, **Scikit-Learn**, and **Streamlit**.

---

## 📌 What It Does

ChurnIQ analyses player behaviour data and predicts **churn probability** (the likelihood a player will stop playing). It provides:

- 📊 **Dataset Overview** — visual stats, distributions, and correlation heatmaps
- 🧠 **Model Training** — train Logistic Regression or Decision Tree on your data, on-the-fly
- 🔮 **Real-Time Prediction** — enter any player's profile and get an instant churn probability score

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/VanshSharma88/ChurnIQ.git
cd ChurnIQ
```

### 2. (Recommended) Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# OR
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
# If streamlit is on your PATH:
streamlit run app.py

# If not (common on Mac):
python3 -m streamlit run app.py
```

### 5. Open in Browser
The app will automatically open at:
```
http://localhost:8501
```

---

## 📂 Dataset

The app uses **`online_gaming_behavior_dataset.csv`** — place it in the `data/` folder or upload it directly in the app.

The dataset has **no explicit Churn column**, so one is derived automatically:

| EngagementLevel | Churn Label |
|---|---|
| `Low` | **1 — At Risk (Churned)** |
| `Medium` or `High` | **0 — Retained** |

### Required Columns
Your CSV **must** include these columns:

| Column | Type | Description |
|---|---|---|
| `PlayerID` | ID | Unique player identifier |
| `Age` | Numeric | Player's age |
| `Gender` | Category | Male / Female |
| `Location` | Category | Country or region |
| `GameGenre` | Category | Action, RPG, Sports, etc. |
| `PlayTimeHours` | Numeric | Total hours played |
| `InGamePurchases` | Binary | 0 = No, 1 = Yes |
| `GameDifficulty` | Category | Easy / Medium / Hard |
| `SessionsPerWeek` | Numeric | Play sessions per week |
| `AvgSessionDurationMinutes` | Numeric | Average session length (minutes) |
| `PlayerLevel` | Numeric | Current in-game level |
| `AchievementsUnlocked` | Numeric | Total achievements earned |
| `EngagementLevel` | **Target** | Low / Medium / High |

---

## �️ Project Structure

```
ChurnIQ/
│
├── app.py                   ← Main Streamlit dashboard (UI + routing)
├── requirements.txt         ← All Python dependencies
├── README.md
│
├── data/
│   └── online_gaming_behavior_dataset.csv
│
└── src/
    ├── __init__.py
    ├── data_loader.py       ← Loads CSV, derives 'Churn' column, defines features
    ├── pipeline.py          ← Scikit-Learn preprocessing + ML pipeline
    └── evaluation.py        ← Metrics (Accuracy, Precision, Recall, AUC) + Confusion Matrix
```

---

## 🧪 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.30 | Web dashboard UI |
| `pandas` | ≥1.5 | Data loading & manipulation |
| `numpy` | ≥1.24 | Numerical operations |
| `scikit-learn` | ≥1.3 | ML models & preprocessing pipelines |
| `matplotlib` | ≥3.7 | Charting |
| `seaborn` | ≥0.12 | Statistical visualisations |

---

## 🔧 How the ML Pipeline Works

```
CSV Upload
    │
    ▼
data_loader.py  →  Derives Churn column  →  Defines feature lists
    │
    ▼
pipeline.py     →  StandardScaler (numeric) + OneHotEncoder (categorical)
    │
    ▼
ML Model        →  Logistic Regression  OR  Decision Tree (max_depth=5)
    │
    ▼
evaluation.py   →  Accuracy, Precision, Recall, AUC-ROC, Confusion Matrix
    │
    ▼
Streamlit UI    →  Real-time churn probability for any player profile
```

---

## 📊 Dashboard Features

### Tab 1 — Dataset Overview
- Stat cards: Total players, features, at-risk count, churn rate %
- Interactive data table preview
- Charts: Churn donut, PlayTime histogram, Sessions/Week histogram, Player Level histogram
- Pearson correlation heatmap across all numeric features

### Tab 2 — Model Training
- Choose algorithm: **Logistic Regression** or **Decision Tree**
- Adjust train/test split (10% – 40%)
- View: Accuracy, Precision, Recall, AUC-ROC metric cards
- Confusion Matrix with guide (True/False Positive/Negative)

### Tab 3 — Predict Churn
- Input a player profile using form fields
- Get an instant **churn probability %**
- Result: 🔴 High Risk or 🟢 Low Risk card with retention advice

---

## ⚠️ Troubleshooting

| Issue | Fix |
|---|---|
| `zsh: command not found: streamlit` | Use `python3 -m streamlit run app.py` instead |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Upload error: `EngagementLevel` missing | Ensure your CSV has an `EngagementLevel` column with values: Low / Medium / High |
| Port 8501 already in use | Run `python3 -m streamlit run app.py --server.port 8502` |
| Blank page in browser | Wait 5 seconds and refresh, or check terminal for errors |

---

## 📄 Report

A full **LaTeX project report** is included:
- 📝 [`ChurnIQ_Report.tex`](./ChurnIQ_Report.tex) — compile with [Overleaf](https://overleaf.com) or `pdflatex`

---

## 👤 Author

**Vansh Sharma**
GitHub: [@VanshSharma88](https://github.com/VanshSharma88)
Project: `Project_14_GenAI` — Milestone 1 (Mid-Semester)
