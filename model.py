
import pandas as pd
import os
import sys
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_score, recall_score

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(__file__))
from preprocessor import scale_features, prepare_data

def train_model(X_train, y_train):
    """
    Trains an XGBoost Classifier.
    """
    print("Training XGBoost Classifier...")
    # Scale_pos_weight is useful for imbalanced datasets, but we already used SMOTE.
    # However, XGBoost creates a new model, so we can just fit it.
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

def calculate_cost(y_true, y_pred, cost_fn=100, cost_fp=1):
    """
    Calculates the detailed cost of error.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        cost_fn: Cost of a False Negative (Fraud missed).
        cost_fp: Cost of a False Positive (Legit transaction blocked).
        
    Returns:
        total_cost: Total calculated cost.
    """
    cm = confusion_matrix(y_true, y_pred)
    # confusion_matrix returns [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    
    print("\n--- Cost of Error Analysis ---")
    print(f"False Negatives (Fraud Missed): {fn}")
    print(f"False Positives (False Alarm): {fp}")
    print(f"Cost per FN: ${cost_fn}")
    print(f"Cost per FP: ${cost_fp}")
    print(f"Total Cost: ${total_cost}")
    
    return total_cost

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data using Precision and Recall.
    """
    print("\nEvaluating Model...")
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Calculate Cost
    calculate_cost(y_test, y_pred)

if __name__ == "__main__":
    # Path to the dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'creditcard.csv')
    
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found at {data_path}")
            
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # 1. Preprocessing
        print("Preprocessing data...")
        df_scaled = scale_features(df)
        
        X = df_scaled.drop('Class', axis=1)
        y = df_scaled['Class']
        
        # 2. Split and Resample
        # prepare_data handles the split and SMOTE-Tomek internally
        X_train_res, X_test, y_train_res, y_test = prepare_data(X, y)
        
        # 3. Train
        model = train_model(X_train_res, y_train_res)
        
        # 4. Evaluate
        evaluate_model(model, X_test, y_test)
        
    except ImportError as e:
        print(f"Import Error: {e}. Please ensure all requirements are installed.")
    except Exception as e:
        print(f"An error occurred: {e}")
