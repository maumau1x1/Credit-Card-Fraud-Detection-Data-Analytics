
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from typing import Tuple
import os

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales 'Amount' and 'Time' columns using RobustScaler.
    Banks care about this because unscaled data leads to biased models.
    
    Args:
        df (pd.DataFrame): Input dataframe containing 'Amount' and 'Time' columns.
        
    Returns:
        pd.DataFrame: Dataframe with scaled columns.
    """
    # Create a copy to avoid modifying the original dataframe
    df_scaled = df.copy()
    
    # Initialize the scaler
    scaler = RobustScaler()
    
    # Scale 'Amount' and 'Time' columns
    cols_to_scale = ['Amount', 'Time']
    
    # Check if columns exist before scaling
    existing_cols = [col for col in cols_to_scale if col in df_scaled.columns]
    
    if existing_cols:
        df_scaled[existing_cols] = scaler.fit_transform(df_scaled[existing_cols])
        
    return df_scaled

def prepare_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits data into train/test sets and applies SMOTE-Tomek resampling to the training set.
    
    Strategy:
    1. Split data into train and test BEFORE applying SMOTE to avoid data leakage.
    2. Use SMOTE-Tomek (a hybrid method). It creates synthetic fraud cases (SMOTE) 
       and then removes "noisy" overlapping points (Tomek Links).
       This shows understanding of advanced data cleaning.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Random state for reproducibility.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
        X_train_resampled, X_test, y_train_resampled, y_test
    """
    # 1. Split your data into train and test before applying SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize SMOTE-Tomek
    print("Applying SMOTE-Tomek resampling... This may take a while.")
    resampler = SMOTETomek(random_state=random_state)
    
    # Apply resampling ONLY to the training data
    X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test

if __name__ == "__main__":
    # Path to the dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'creditcard.csv')
    
    try:
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"Data loaded. Shape: {df.shape}")
        
        # 1. Scale Features
        print("Scaling features...")
        df_scaled = scale_features(df)
        
        # Separate Features and Target
        # Assuming 'Class' is the target column based on standard credit card fraud datasets
        if 'Class' not in df_scaled.columns:
            raise ValueError("Target column 'Class' not found in dataset.")
            
        X = df_scaled.drop('Class', axis=1)
        y = df_scaled['Class']
        
        # 2. Split and Resample
        print("Splitting and resampling data...")
        X_train_res, X_test, y_train_res, y_test = prepare_data(X, y)
        
        print("\nPreprocessing Complete!")
        print("-" * 30)
        print(f"Original Training Set Shape: {len(y) * 0.8} samples (approx)") # 80% split
        print(f"Resampled Training Set Shape: {X_train_res.shape}")
        print(f"Test Set Shape: {X_test.shape}")
        print("-" * 30)
        print("Class Distribution in Resampled Training Set:")
        print(y_train_res.value_counts())
        
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")
