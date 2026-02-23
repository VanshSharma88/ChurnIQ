"""
pipeline.py
-----------
This file builds a Scikit-Learn "Pipeline" — a chain of steps that:
  Step 1: Preprocesses the data (cleans, scales, and encodes it)
  Step 2: Trains a machine learning model on the processed data

Why a Pipeline? So that preprocessing + model training happen in one clean step,
avoiding errors like accidentally scaling test data with training stats.
"""

# Pipeline chains multiple steps together (like an assembly line)
from sklearn.pipeline import Pipeline

# StandardScaler: scales numbers so they're all on a similar range (mean=0, std=1)
# OneHotEncoder: converts text categories into numbers (e.g., 'Male' → [1, 0])
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ColumnTransformer: applies different transformations to different columns
from sklearn.compose import ColumnTransformer

# Logistic Regression: a simple, fast, interpretable classification model
# (despite the name, it predicts categories, not continuous values)
from sklearn.linear_model import LogisticRegression

# Decision Tree: splits data into yes/no branches to make predictions
# (very interpretable — you can visualise the actual decision rules)
from sklearn.tree import DecisionTreeClassifier


def create_pipeline(numerical_features, categorical_features, model_type='LogisticRegression'):
    """
    Builds and returns a full ML pipeline.

    Parameters:
    - numerical_features  : list of column names that contain numbers
    - categorical_features: list of column names that contain text/categories
    - model_type          : which algorithm to use ('LogisticRegression' or 'DecisionTree')

    Returns:
    - A Scikit-Learn Pipeline object (ready to be trained with .fit())
    """

    # ── Step 1a: Preprocessing for NUMBER columns ──────────────────────
    # StandardScaler makes all numbers comparable.
    # Example: Age (15–49) and PlayTimeHours (0–24) are on different scales.
    # After scaling, both will be centred around 0.
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())   # Scale numbers to mean=0, std=1
    ])

    # ── Step 1b: Preprocessing for TEXT/CATEGORY columns ───────────────
    # OneHotEncoder converts categories into binary (0/1) columns.
    # Example: Gender ['Male', 'Female'] → Gender_Male: [1,0], Gender_Female: [0,1]
    # handle_unknown='ignore' means: if the model sees a new category at prediction time,
    # it won't crash — it will just treat it as all zeros.
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ── Step 1c: Combine both transformers using ColumnTransformer ──────
    # This applies numeric_transformer to number columns
    # and categorical_transformer to text columns — simultaneously.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),   # Apply scaler to numbers
            ('cat', categorical_transformer, categorical_features)  # Apply OHE to text
        ]
    )

    # ── Step 2: Choose the ML Model ────────────────────────────────────
    if model_type == 'DecisionTree':
        # Decision Tree with max_depth=5 to prevent overfitting
        # (overfitting = model memorises training data but fails on new data)
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
    else:
        # Default: Logistic Regression
        # max_iter=1000 → allow up to 1000 iterations to find the best fit
        # C=1.0 → regularisation strength (higher C = less penalty on complexity)
        # random_state=42 → ensures reproducibility (same result every run)
        model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)

    # ── Step 3: Combine preprocessor + model into one Pipeline ─────────
    # When we call pipeline.fit(X_train, y_train):
    #   1. Data goes through the preprocessor (scaling + encoding)
    #   2. The processed data goes into the model for training
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Step 1: clean and transform data
        ('model', model)                 # Step 2: train the ML model
    ])

    return clf