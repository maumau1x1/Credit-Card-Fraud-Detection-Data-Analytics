import pandas as pd
import os
import sys
import numpy as np

# Ensure we can import from the src directory
sys.path.append(os.path.dirname(__file__))
from preprocessor import scale_features, prepare_data
from model import train_model

def export_for_powerbi():
    """
    Loads data, trains the model, generates predictions, and exports a CSV for Power BI.
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'creditcard.csv')
    export_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'powerbi_export.csv')
    
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Source data not found at {data_path}")
            
        print(f"Loading data from {data_path}...")
        df_original = pd.read_csv(data_path)
        
        # 1. Preprocessing for model
        print("Preprocessing data for model training...")
        df_scaled = scale_features(df_original)
        
        X = df_scaled.drop('Class', axis=1)
        y = df_scaled['Class']
        
        # 2. Split and Resample to train a good model
        print("Preparing training and test sets...")
        X_train_res, X_test, y_train_res, y_test = prepare_data(X, y)
        
        # 3. Train Model
        print("Training model...")
        model = train_model(X_train_res, y_train_res)
        
        # 4. Generate Predictions for the ENTIRE dataset (for the dashboard)
        print("Generating predictions for the entire dataset...")
        # We use the scaled features for prediction
        X_all = df_scaled.drop('Class', axis=1)
        probabilities = model.predict_proba(X_all)[:, 1]
        predictions = model.predict(X_all)
        
        # 5. Create Export Dataframe
        # Start with original unscaled Time and Amount for better dashboard visuals
        export_df = df_original.copy()
        
        export_df['Predicted_Class'] = predictions
        export_df['Fraud_Probability'] = probabilities
        
        # 6. Define Result Types (TP, TN, FP, FN)
        print("Categorizing results...")
        def categorize(row):
            actual = row['Class']
            pred = row['Predicted_Class']
            if actual == 1 and pred == 1: return 'True Positive (Fraud Detected)'
            if actual == 0 and pred == 0: return 'True Negative (Legit Checked)'
            if actual == 0 and pred == 1: return 'False Positive (False Alarm)'
            if actual == 1 and pred == 0: return 'False Negative (Fraud Missed)'
            return 'Unknown'

        export_df['Result_Type'] = export_df.apply(categorize, axis=1)
        
        # 7. Export to CSV
        print(f"Exporting data to {export_path}...")
        export_df.to_csv(export_path, index=False)
        
        print("\nExport Complete!")
        print(f"Total rows: {len(export_df)}")
        print(f"Fraud detected (TP): {len(export_df[export_df['Result_Type'] == 'True Positive (Fraud Detected)'])}")
        print(f"False Alarms (FP): {len(export_df[export_df['Result_Type'] == 'False Positive (False Alarm)'])}")
        print(f"Missed Fraud (FN): {len(export_df[export_df['Result_Type'] == 'False Negative (Fraud Missed)'])}")
        
    except Exception as e:
        print(f"An error occurred during export: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_for_powerbi()
