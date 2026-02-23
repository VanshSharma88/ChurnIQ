# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# matplotlib and seaborn for creating charts/plots
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn metrics: functions that calculate how good our model is
from sklearn.metrics import (
    accuracy_score,    # % of correct predictions
    precision_score,   # of predicted churners, how many were right
    recall_score,      # of actual churners, how many did we catch
    roc_auc_score,     # area under the ROC curve (model's overall discrimination ability)
    confusion_matrix   # table showing correct vs incorrect predictions
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluates how well the trained model performs on unseen test data.

    Parameters:
    - model  : the trained Scikit-Learn pipeline
    - X_test : the test features (input data the model hasn't seen)
    - y_test : the actual true labels (0 = retained, 1 = churned)

    Returns:
    - metrics : a dictionary with Accuracy, Precision, Recall, AUC scores
    - y_pred  : the model's predicted labels (used for confusion matrix)
    """

    # Ask the model to make predictions on the test set
    # y_pred contains 0s and 1s (predicted class labels)
    y_pred = model.predict(X_test)

    # predict_proba gives the probability of each class
    # [:, 1] means "give me the probability of class 1 (Churn)"
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate all 4 evaluation metrics and store them in a dictionary
    metrics = {
        'Accuracy':  accuracy_score(y_test, y_pred),   # Overall correctness
        'Precision': precision_score(y_test, y_pred),  # Quality of churn predictions
        'Recall':    recall_score(y_test, y_pred),     # How many churners we caught
        'AUC':       roc_auc_score(y_test, y_prob)     # Overall discrimination ability
    }

    return metrics, y_pred


def plot_confusion_matrix(y_test, y_pred):
    """
    Creates a heatmap showing the Confusion Matrix.

    A Confusion Matrix is a 2x2 table:
    ┌───────────────────┬────────────────────┐
    │  True Negative    │  False Positive    │
    │  (correctly said  │  (wrongly said     │
    │   player stayed)  │   player churned)  │
    ├───────────────────┼────────────────────┤
    │  False Negative   │  True Positive     │
    │  (missed a        │  (correctly caught │
    │   churner)        │   a churner)       │
    └───────────────────┴────────────────────┘

    Parameters:
    - y_test : the actual true labels
    - y_pred : the model's predicted labels

    Returns:
    - fig : a matplotlib figure object (Streamlit can display this with st.pyplot)
    """

    # Compute the confusion matrix values
    cm = confusion_matrix(y_test, y_pred)

    # Create a new figure and axes for the plot
    # figsize=(6, 4) means 6 inches wide, 4 inches tall
    fig, ax = plt.subplots(figsize=(6, 4))

    # Draw the heatmap using seaborn
    # annot=True     → show the numbers inside each cell
    # fmt='d'        → format numbers as integers (no decimals)
    # cmap='Blues'   → use a blue colour gradient
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

    # Label the axes clearly
    ax.set_xlabel('Predicted Label')   # What the model guessed
    ax.set_ylabel('Actual Label')      # What the truth was
    ax.set_title('Confusion Matrix')   # Chart title

    return fig