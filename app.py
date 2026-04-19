"""
app.py — Main Streamlit application for ChurnIQ: Player Churn Prediction
Run with: python3 -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os

from src.data_loader import load_data, get_feature_lists
from src.pipeline import create_pipeline
from src.evaluation import evaluate_model, plot_confusion_matrix
from src.agent import build_agent_graph

# Load environment variables
load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ – Player Churn Prediction",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
p, span, div, label, li {
    font-size: 15px !important;
    line-height: 1.65 !important;
}

/* Streamlit overrides */
[data-testid="stDataFrame"] * { font-size: 13px !important; }
[data-testid="stButton"] button {
    font-size: 15px !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #C4B5FD !important;
}
div[data-baseweb="tab-list"] {
    background: #16213E;
    border-radius: 14px;
    padding: 5px;
    gap: 4px;
}
div[data-baseweb="tab"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
}

/* Hero */
.hero-title {
    font-size: 3.8rem !important;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF 0%, #A78BFA 50%, #60A5FA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.6rem;
}
.hero-sub {
    font-size: 1.15rem !important;
    color: #9CA3AF;
    margin-bottom: 0.3rem;
    line-height: 1.6;
}

/* Upload zone */
.upload-zone {
    background: linear-gradient(160deg, #1A1A2E 0%, #16213E 100%);
    border: 2px dashed #4F46E5;
    border-radius: 22px;
    padding: 3rem 2rem;
    text-align: center;
    box-shadow: 0 10px 40px rgba(108, 99, 255, 0.18);
    transition: border-color 0.3s;
}
.upload-zone:hover { border-color: #818CF8; }

/* Schema table */
.schema-table {
    width: 100%;
    border-collapse: collapse;
    border-radius: 14px;
    overflow: hidden;
}
.schema-table th {
    background: #1E1B4B;
    color: #A78BFA;
    font-size: 13px !important;
    font-weight: 700;
    padding: 10px 16px;
    text-align: left;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.schema-table tr {
    border-bottom: 1px solid #2D2D4E;
    transition: background 0.15s;
}
.schema-table tr:hover { background: #1A1A2E; }
.schema-table td {
    padding: 10px 16px;
    font-size: 14px !important;
    color: #D1D5DB;
    white-space: nowrap;
}
.schema-table td:first-child {
    color: #A78BFA;
    font-weight: 600;
    font-family: monospace;
    font-size: 13px !important;
}
.target-badge {
    background: linear-gradient(135deg, #6C63FF33, #60A5FA33);
    border: 1px solid #6C63FF99;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px !important;
    color: #A78BFA;
    margin-left: 8px;
    white-space: nowrap;
}

/* Stat cards */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
    margin: 1rem 0;
}
.stat-card {
    background: linear-gradient(145deg, #1A1A2E, #16213E);
    border: 1px solid #2D2D4E;
    border-radius: 16px;
    padding: 1.4rem 1rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(108,99,255,0.2);
}
.stat-num {
    font-size: 2rem !important;
    font-weight: 800;
    background: linear-gradient(135deg, #818CF8, #60A5FA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.stat-lbl {
    font-size: 12px !important;
    color: #9CA3AF;
    margin-top: 5px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Section header */
.sec-hdr {
    font-size: 1.25rem !important;
    font-weight: 700;
    color: #E2E8F0;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-hdr::after {
    content: '';
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, #6C63FF55, transparent);
    border-radius: 2px;
}

/* Metric pills */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 1rem 0;
}
.metric-pill {
    background: linear-gradient(145deg, #1E1B4B, #1A1A2E);
    border: 1px solid #4F46E566;
    border-radius: 16px;
    padding: 1.6rem 1rem;
    text-align: center;
}
.metric-val {
    font-size: 2.2rem !important;
    font-weight: 800;
    color: #A78BFA;
    line-height: 1.2;
}
.metric-lbl {
    font-size: 12px !important;
    color: #9CA3AF;
    margin-top: 6px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Churn result cards */
.result-high {
    background: linear-gradient(145deg, #450A0A, #1A0505);
    border: 1px solid #EF4444AA;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
}
.result-low {
    background: linear-gradient(145deg, #052E16, #031A0D);
    border: 1px solid #10B981AA;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
}

/* Info box */
.info-box {
    background: #1E1B4B;
    border: 1px solid #4F46E5;
    border-left: 4px solid #818CF8;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    font-size: 14px !important;
    color: #C4B5FD;
    margin: 0.5rem 0;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

if 'df' not in st.session_state:
    st.session_state['df'] = None



# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
def show_landing():
    # ── Sidebar for API Key ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('### 🔑 Agent Configuration')
        groq_key = st.text_input("Groq API Key (Optional if loaded from .env)", type="password")
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
        st.markdown('---')
        st.markdown('This End-Sem feature uses LangGraph to generate an Agentic Retrieval-Augmented Engagement strategy.')
        
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding: 2rem 0 1rem 0;">
        <div class="hero-title">🎮 ChurnIQ</div>
        <div class="hero-sub">Intelligent Player Churn Prediction — Powered by Traditional Machine Learning</div>
        <div style="height:2px; width:120px;
             background:linear-gradient(90deg,#6C63FF,#60A5FA);
             border-radius:2px; margin-top:1rem;"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload + Schema in two columns ────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:3.5rem; margin-bottom:1rem;">📂</div>
            <div style="font-size:1.3rem; font-weight:700; color:#E2E8F0; margin-bottom:0.6rem;">
                Upload Your Dataset
            </div>
            <div style="font-size:14px; color:#9CA3AF; line-height:1.7;">
                Upload a <strong style="color:#A78BFA">.csv</strong> file containing player behaviour data.<br>
                Must include an <code style="background:#2D2D4E; padding:2px 6px; border-radius:4px;">EngagementLevel</code>
                column with values:<br>
                <span style="color:#10B981; font-weight:600;">High</span> &nbsp;·&nbsp;
                <span style="color:#F59E0B; font-weight:600;">Medium</span> &nbsp;·&nbsp;
                <span style="color:#EF4444; font-weight:600;">Low</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop CSV here or click to browse",
            type=["csv"],
            label_visibility="collapsed"
        )
        if uploaded:
            with st.spinner("🔄 Loading and processing dataset…"):
                df = load_data(uploaded)
            if df is not None:
                st.session_state['df'] = df
                st.success(f"✅ Loaded **{df.shape[0]:,} rows** and **{df.shape[1]} columns** successfully!")
                st.rerun()
            else:
                st.error("❌ Upload failed. Ensure the CSV has an `EngagementLevel` column.")

    with right:
        st.markdown('<div class="sec-hdr">📋 Expected Dataset Schema</div>', unsafe_allow_html=True)
        schema = [
            ("PlayerID",                   "Unique player identifier",            ""),
            ("Age",                        "Player age",                          "numeric"),
            ("Gender",                     "Male / Female",                       "category"),
            ("Location",                   "Country / Region",                    "category"),
            ("GameGenre",                  "Action, RPG, Sports…",                "category"),
            ("PlayTimeHours",              "Total hours played",                  "numeric"),
            ("InGamePurchases",            "Did player purchase? (0 = No, 1 = Yes)", "numeric"),
            ("GameDifficulty",             "Easy / Medium / Hard",                "category"),
            ("SessionsPerWeek",            "Sessions played per week",            "numeric"),
            ("AvgSessionDurationMinutes",  "Average session length in minutes",   "numeric"),
            ("PlayerLevel",                "Current in-game level",               "numeric"),
            ("AchievementsUnlocked",       "Total achievements earned",           "numeric"),
            ("EngagementLevel",            "Low / Medium / High",                 "🎯 target"),
        ]
        rows_html = ""
        for col_name, desc, dtype in schema:
            badge = ""
            if dtype == "🎯 target":
                badge = '<span class="target-badge">🎯 TARGET</span>'
            elif dtype:
                badge = f'<span style="font-size:11px;color:#6B7280;background:#1E293B;padding:2px 7px;border-radius:4px;margin-left:6px;">{dtype}</span>'
            rows_html += f"""
            <tr>
                <td>{col_name}</td>
                <td>{desc}{badge}</td>
            </tr>"""

        st.markdown(f"""
        <div style="border:1px solid #2D2D4E; border-radius:14px; overflow:hidden;">
        <table class="schema-table">
            <thead>
                <tr>
                    <th style="width:240px;">Column</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def show_dashboard(df):
    numerical_features, categorical_features = get_feature_lists()

    # ── Top bar ───────────────────────────────────────────────────────────────
    col_title, col_btn = st.columns([5, 1])
    with col_title:
        st.markdown('<div class="hero-title" style="font-size:2.4rem;">🎮 ChurnIQ &nbsp;<span style="font-weight:300;color:#6B7280;font-size:1.2rem;">Dashboard</span></div>',
                    unsafe_allow_html=True)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄  New Dataset", use_container_width=True):
            st.session_state['df'] = None
            st.session_state.pop('pipeline', None)
            st.rerun()

    st.markdown("<hr style='border-color:#2D2D4E; margin:0.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊   Dataset Overview", "🧠   Model Training", "🔮   Predict Churn"])

    # ════════════════════════════════════════════════════
    # TAB 1 — DATASET OVERVIEW
    # ════════════════════════════════════════════════════
    with tab1:
        churn_count  = int(df['Churn'].sum())
        retain_count = int((df['Churn'] == 0).sum())
        churn_pct    = churn_count / len(df) * 100

        # Stat grid
        st.markdown('<div class="sec-hdr">🔍 At a Glance</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-num">{df.shape[0]:,}</div>
                <div class="stat-lbl">Total Players</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{df.shape[1]}</div>
                <div class="stat-lbl">Features</div>
            </div>
            <div class="stat-card">
                <div class="stat-num" style="color:#F87171;">{churn_count:,}</div>
                <div class="stat-lbl">At-Risk (Churn)</div>
            </div>
            <div class="stat-card">
                <div class="stat-num" style="color:#34D399;">{retain_count:,}</div>
                <div class="stat-lbl">Retained</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{churn_pct:.1f}%</div>
                <div class="stat-lbl">Churn Rate</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Data preview
        st.markdown('<div class="sec-hdr">📄 Data Preview <span style="font-weight:400;color:#6B7280;font-size:0.85rem;">(first 10 rows)</span></div>',
                    unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        # Charts
        st.markdown('<div class="sec-hdr">📈 Distributions</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")

        def chart_style(ax):
            ax.set_facecolor('#0F0F1A')
            ax.tick_params(colors='#9CA3AF', labelsize=11)
            for spine in ax.spines.values():
                spine.set_edgecolor('#2D2D4E')

        # Donut chart
        with c1:
            fig, ax = plt.subplots(figsize=(5, 4.5), facecolor='#0F0F1A')
            ax.set_facecolor('#0F0F1A')
            wedges, _ = ax.pie(
                [retain_count, churn_count],
                colors=['#10B981', '#EF4444'],
                startangle=90,
                wedgeprops=dict(width=0.52, edgecolor='#0F0F1A', linewidth=2.5)
            )
            ax.text(0, 0.05, f"{churn_pct:.1f}%", ha='center', va='center',
                    fontsize=20, fontweight='bold', color='#E2E8F0')
            ax.text(0, -0.22, "Churn", ha='center', va='center',
                    fontsize=12, color='#9CA3AF')
            patches = [mpatches.Patch(color='#10B981', label=f'Retained  ({retain_count:,})'),
                       mpatches.Patch(color='#EF4444', label=f'Churn  ({churn_count:,})')]
            ax.legend(handles=patches, loc='lower center', ncol=2, frameon=False,
                      labelcolor='#9CA3AF', fontsize=12, bbox_to_anchor=(0.5, -0.08))
            ax.set_title("Churn Distribution", color='#E2E8F0', fontsize=15, fontweight='600', pad=14)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Play time
        with c2:
            fig2, ax2 = plt.subplots(figsize=(5, 4.5), facecolor='#0F0F1A')
            chart_style(ax2)
            ax2.hist(df['PlayTimeHours'], bins=30, color='#6C63FF', edgecolor='#0F0F1A', alpha=0.9)
            ax2.set_title("PlayTime Distribution", color='#E2E8F0', fontsize=15, fontweight='600')
            ax2.set_xlabel("Hours Played", color='#9CA3AF', fontsize=12)
            ax2.set_ylabel("Number of Players", color='#9CA3AF', fontsize=12)
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        c3, c4 = st.columns(2, gap="medium")
        with c3:
            fig3, ax3 = plt.subplots(figsize=(5, 4), facecolor='#0F0F1A')
            chart_style(ax3)
            ax3.hist(df['SessionsPerWeek'], bins=20, color='#60A5FA', edgecolor='#0F0F1A', alpha=0.9)
            ax3.set_title("Sessions Per Week", color='#E2E8F0', fontsize=15, fontweight='600')
            ax3.set_xlabel("Sessions", color='#9CA3AF', fontsize=12)
            ax3.set_ylabel("Number of Players", color='#9CA3AF', fontsize=12)
            fig3.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

        with c4:
            fig5, ax5 = plt.subplots(figsize=(5, 4), facecolor='#0F0F1A')
            chart_style(ax5)
            ax5.hist(df['PlayerLevel'], bins=25, color='#F59E0B', edgecolor='#0F0F1A', alpha=0.9)
            ax5.set_title("Player Level Distribution", color='#E2E8F0', fontsize=15, fontweight='600')
            ax5.set_xlabel("Player Level", color='#9CA3AF', fontsize=12)
            ax5.set_ylabel("Number of Players", color='#9CA3AF', fontsize=12)
            fig5.tight_layout()
            st.pyplot(fig5, use_container_width=True)
            plt.close(fig5)

        # Correlation heatmap
        st.markdown('<div class="sec-hdr">🔗 Feature Correlation Matrix</div>', unsafe_allow_html=True)
        num_df = df[numerical_features + ['Churn']].corr()
        fig4, ax4 = plt.subplots(figsize=(12, 5), facecolor='#0F0F1A')
        ax4.set_facecolor('#0F0F1A')
        sns.heatmap(num_df, annot=True, fmt='.2f', cmap='coolwarm',
                    linewidths=0.6, linecolor='#1E293B',
                    annot_kws={"size": 11, "color": "#E2E8F0", "weight": "500"},
                    ax=ax4, cbar_kws={"shrink": 0.8})
        ax4.tick_params(colors='#9CA3AF', labelsize=11)
        ax4.set_title("Pearson Correlation between Features and Churn",
                      color='#E2E8F0', fontsize=13, fontweight='600', pad=14)
        fig4.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

        # Column info
        st.markdown('<div class="sec-hdr">🗂 Column Summary</div>', unsafe_allow_html=True)
        info_df = pd.DataFrame({
            'Column':        df.columns,
            'Data Type':     df.dtypes.astype(str).values,
            'Non-Null Count': df.notnull().sum().values,
            'Null Count':    df.isnull().sum().values,
            'Unique Values': df.nunique().values
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════
    # TAB 2 — MODEL TRAINING
    # ════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="sec-hdr">⚙️ Configure & Train</div>', unsafe_allow_html=True)

        cfg1, cfg2, cfg3 = st.columns([2, 2, 1], gap="medium")
        with cfg1:
            model_type = st.selectbox(
                "🤖 Algorithm",
                ["LogisticRegression", "DecisionTree", "RandomForest"],
                help="LogisticRegression: fast linear model · DecisionTree: interpretable rule-based model · RandomForest: powerful ensemble model"
            )
        with cfg2:
            test_size = st.slider("🔀 Test Split Size", 0.10, 0.40, 0.20, 0.05,
                                  help="Fraction of data held out for testing")
        with cfg3:
            st.markdown("<br>", unsafe_allow_html=True)
            train_btn = st.button("🚀 Train", use_container_width=True)

        # Algorithm info
        algo_desc = {
            "LogisticRegression": "Finds a linear decision boundary separating churners from retained players. Fast, explainable, and works well on linearly separable data.",
            "DecisionTree": "Builds a tree of yes/no rules (max depth = 5) to classify players. Highly interpretable — great for explaining predictions.",
            "RandomForest": "Builds an ensemble of multiple decision trees to improve accuracy and prevent overfitting. Powerful and robust, standard for ML."
        }
        st.markdown(f'<div class="info-box">💡 <strong>{model_type}</strong> — {algo_desc[model_type]}</div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if train_btn:
            with st.spinner(f"⏳ Training {model_type}…"):
                X = df.drop(['PlayerID', 'Churn', 'EngagementLevel'], axis=1, errors='ignore')
                y = df['Churn']
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                pipeline = create_pipeline(numerical_features, categorical_features, model_type=model_type)
                pipeline.fit(X_train, y_train)
                metrics, y_pred = evaluate_model(pipeline, X_test, y_test)
                st.session_state['pipeline'] = pipeline

            st.success(f"✅ {model_type} trained on **{len(X_train):,}** samples · tested on **{len(X_test):,}** samples")

            # Metrics
            st.markdown('<div class="sec-hdr">📊 Evaluation Results</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-pill">
                    <div class="metric-val">{metrics['Accuracy']:.1%}</div>
                    <div class="metric-lbl">Accuracy</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-val">{metrics['Precision']:.1%}</div>
                    <div class="metric-lbl">Precision</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-val">{metrics['Recall']:.1%}</div>
                    <div class="metric-lbl">Recall</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-val">{metrics['AUC']:.1%}</div>
                    <div class="metric-lbl">AUC-ROC</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confusion matrix
            st.markdown('<div class="sec-hdr">🟪 Confusion Matrix</div>', unsafe_allow_html=True)
            cm_col, guide_col = st.columns([1, 1], gap="large")
            with cm_col:
                fig_cm = plot_confusion_matrix(y_test, y_pred)
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)
            with guide_col:
                st.markdown("""
                <div class="info-box" style="margin-top:1rem; line-height:2;">
                    <strong>How to read the matrix</strong><br>
                    🟦 <strong>True Negative</strong> — Correctly predicted player stayed<br>
                    🟥 <strong>False Positive</strong> — Predicted churn, player actually stayed<br>
                    🟧 <strong>False Negative</strong> — Missed a churner (most costly miss)<br>
                    🟩 <strong>True Positive</strong> — Correctly caught a churner<br><br>
                    A high <strong>Recall</strong> means fewer missed churners.
                </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # TAB 3 — PREDICTION
    # ════════════════════════════════════════════════════
    with tab3:
        if 'pipeline' not in st.session_state:
            st.markdown("""
            <div class="info-box" style="text-align:center; padding:2rem;">
                ⚠️ &nbsp; Train a model first in the <strong>Model Training</strong> tab before predicting.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="sec-hdr">🎯 Player Profile Input</div>', unsafe_allow_html=True)
            st.markdown("<div style='color:#9CA3AF; font-size:14px; margin-bottom:1rem;'>Fill in the player's details below to get a real-time churn probability prediction.</div>",
                        unsafe_allow_html=True)

            with st.form("prediction_form"):
                r1c1, r1c2, r1c3, r1c4 = st.columns(4, gap="medium")
                with r1c1:
                    age       = st.number_input("🎂 Age", 10, 100, 25)
                    gender    = st.selectbox("👤 Gender", df['Gender'].unique())
                with r1c2:
                    location  = st.selectbox("🌍 Location", df['Location'].unique())
                    genre     = st.selectbox("🎮 Game Genre", df['GameGenre'].unique())
                with r1c3:
                    playtime  = st.number_input("⏱ PlayTime (Hours)", 0.0, 50.0, 10.0, step=0.5)
                    sessions  = st.number_input("📅 Sessions/Week", 0, 30, 5)
                with r1c4:
                    avg_dur   = st.number_input("🕐 Avg Session (Min)", 0, 300, 30)
                    level     = st.number_input("⭐ Player Level", 1, 200, 10)

                r2c1, r2c2, r2c3, _ = st.columns(4, gap="medium")
                with r2c1:
                    difficulty  = st.selectbox("🎯 Difficulty", df['GameDifficulty'].unique())
                with r2c2:
                    achievements = st.number_input("🏆 Achievements", 0, 200, 5)
                with r2c3:
                    purchases = st.selectbox("💳 In-Game Purchases", [0, 1],
                                             format_func=lambda x: "Yes (1)" if x else "No (0)")

                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("🔮  Predict Churn Probability", use_container_width=True)

            if submitted:
                input_df = pd.DataFrame({
                    'Age': [age], 'Gender': [gender], 'Location': [location],
                    'GameGenre': [genre], 'PlayTimeHours': [playtime],
                    'InGamePurchases': [purchases], 'GameDifficulty': [difficulty],
                    'SessionsPerWeek': [sessions], 'AvgSessionDurationMinutes': [avg_dur],
                    'PlayerLevel': [level], 'AchievementsUnlocked': [achievements]
                })
                pipe = st.session_state['pipeline']
                
                # Setup Agent
                agent = build_agent_graph(pipe)
                
                initial_state = {
                    "player_data": input_df.iloc[0].to_dict(),
                    "churn_proba": 0.0,
                    "is_churn": False,
                    "retrieved_strategies": [],
                    "structured_evaluation": {},
                    "error": ""
                }
                
                st.markdown("<br><hr>", unsafe_allow_html=True)
                st.markdown("### 🤖 Agent Execution Log")
                
                # Run the LangGraph
                progress_container = st.empty()
                with progress_container.container():
                    st.info("🧠 Initializing agent risk prediction...")
                    
                final_state = initial_state
                for event in agent.stream(initial_state):
                    for key, value in event.items():
                        if key == "predict_risk":
                            with progress_container.container():
                                proba = value.get("churn_proba", 0)
                                is_churn = value.get("is_churn", False)
                                risk_str = "HIGH" if is_churn else "LOW"
                                stat_color = "red" if is_churn else "green"
                                st.warning(f"🔍 **Risk Predicted:** :{stat_color}[{risk_str} ({proba:.1%})]")
                                if is_churn:
                                    st.info("📚 Consulting FAISS knowledge base for strategies...")
                        elif key == "retrieve_knowledge":
                            with progress_container.container():
                                strats = value.get("retrieved_strategies", [])
                                st.success(f"✅ Extracted {len(strats)} relevant strategies.")
                                st.info("✍️ Synthesizing retention plan using LLM...")
                        elif key == "generate_plan":
                            with progress_container.container():
                                if value.get("error"):
                                    st.error(f"❌ {value['error']}")
                                else:
                                    st.success("✨ Retention blueprint finalized!")
                        final_state = value
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Final output display
                if final_state.get('error'):
                    st.error(f"Execution Error: {final_state['error']}")
                elif final_state.get("is_churn") == False:
                    st.markdown(f"""
                    <div class="result-low">
                        <div style="font-size:3rem; margin-bottom:0.5rem;">✅</div>
                        <div style="font-size:1.5rem; font-weight:800; color:#6EE7B7; margin-bottom:0.5rem;">
                            Low Churn Risk
                        </div>
                        <div style="font-size:3rem; font-weight:800; color:#10B981; line-height:1;">
                            {final_state.get('churn_proba'):.1%}
                        </div>
                        <div style="color:#9CA3AF; font-size:14px; margin-top:1rem;">
                            This player appears highly engaged. Keep up the great experience!
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-high">
                        <div style="font-size:3rem; margin-bottom:0.5rem;">⚠️</div>
                        <div style="font-size:1.5rem; font-weight:800; color:#FCA5A5; margin-bottom:0.5rem;">
                            High Churn Risk
                        </div>
                        <div style="font-size:3rem; font-weight:800; color:#EF4444; line-height:1;">
                            {final_state.get('churn_proba'):.1%}
                        </div>
                    </div>""", unsafe_allow_html=True)
                    
                    eval_data = final_state.get("structured_evaluation", {})
                    st.markdown("### 📝 Agentic Structured Output")
                    
                    with st.container(border=True):
                        st.markdown(f"#### 📊 Summary: Player Behavior Overview\n{eval_data.get('Summary', 'N/A')}")
                        st.markdown("---")
                        st.markdown(f"#### 🧠 Analysis: Churn Risk Interpretation\n{eval_data.get('Analysis', 'N/A')}")
                        st.markdown("---")
                        st.markdown(f"#### 🎯 Plan: Engagement & Retention Recs\n{eval_data.get('Plan', 'N/A')}")
                        st.markdown("---")
                        st.markdown(f"#### 🔗 Refs: Supporting References\n*{eval_data.get('Refs', 'N/A')}*")
                        
                    st.warning(f"**⚠️ Disclaimer:** {eval_data.get('Disclaimer', 'N/A')}")


# ── Router ─────────────────────────────────────────────────────────────────────
if st.session_state['df'] is None:
    show_landing()
else:
    show_dashboard(st.session_state['df'])
