# 🎮 Intelligent Player Churn Prediction

## 📌 Project Overview
This project is a **Machine Learning-based Churn Prediction System** designed to identify players at risk of stopping gameplay. It uses **Traditional ML algorithms** (Random Forest, Logistic Regression) to analyze player behavior and predict churn probability.

**Milestone 1 (Mid-Sem)**:
-   **Goal**: Predict Player Churn using supervised learning.
-   **Tech Stack**: Python, Scikit-Learn, Streamlit, Pandas.
-   **Key Feature**: Real-time prediction dashboard.

## 📂 Dataset & Logic
The system uses the `online_gaming_behavior_dataset.csv`.
Since the dataset does not have an explicit `Churn` column, we **derive the target variable** from `EngagementLevel`:
-   **Churn = 1 (High Risk)**: Players with `EngagementLevel` = 'Low'.
-   **Churn = 0 (Retained)**: Players with `EngagementLevel` = 'Medium' or 'High'.

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone <repository_url>
cd Project_14_GenAI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

## 📊 Features
1.  **Data Analysis**: Visualize player stats and churn distribution.
2.  **Model Training**: Train Random Forest or Logistic Regression models on the fly.
3.  **Prediction System**: Input specific player metrics (PlayTime, Sessions, Level) to get a churn probability score.

## 🛠 Project Structure
-   `app.py`: Main Streamlit dashboard.
-   `src/data_loader.py`: Handles CSV loading and 'Churn' label creation.
-   `src/pipeline.py`: Scikit-Learn pipelines for preprocessing (OneHotEncoding, Scaling).
-   `src/evaluation.py`: Metrics (Accuracy, AUC) and Confusion Matrix potting.


## 📊 Visualizations
