"""
Supervised Model Training Script
=================================
Trains and evaluates three supervised machine learning models:
1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier

Uses Isolation Forest predictions as labels for supervised learning.

Author: Advanced Computer Programming Course
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import STANDARDIZED_DATA_PATH, MODELS_DIR, RANDOM_STATE


def load_data_with_labels():
    """Load standardized data and generate labels using Isolation Forest."""
    print("="*70)
    print("SUPERVISED MODEL TRAINING")
    print("="*70)
    
    print("\n[1] Loading standardized data...")
    try:
        df = pd.read_csv(STANDARDIZED_DATA_PATH)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        print(f"    ✓ Loaded {len(df):,} records")
    except FileNotFoundError:
        print(f"    ✗ Error: Standardized data not found")
        print("    → Run the data pipeline first")
        sys.exit(1)
    
    print("\n[2] Generating labels using Isolation Forest predictions...")
    
    # Load the trained Isolation Forest model
    model_path = os.path.join(MODELS_DIR, 'isolation_forest_model.pkl')
    try:
        iso_forest = joblib.load(model_path)
        print(f"    ✓ Loaded Isolation Forest model")
        
        # Generate predictions (-1 for anomaly, 1 for normal)
        X = df[['Value_Standardized']].values
        predictions = iso_forest.predict(X)
        
        # Convert to binary labels (0 for normal, 1 for anomaly)
        # IF outputs: 1=normal, -1=anomaly
        # We convert to: 0=normal, 1=anomaly
        labels = (predictions == -1).astype(int)
        
        df['Label'] = labels
        
        n_normal = np.sum(labels == 0)
        n_anomaly = np.sum(labels == 1)
        
        print(f"    ✓ Labels generated")
        print(f"      - Normal (0):  {n_normal:,} ({n_normal/len(labels)*100:.2f}%)")
        print(f"      - Anomaly (1): {n_anomaly:,} ({n_anomaly/len(labels)*100:.2f}%)")
        
    except FileNotFoundError:
        print(f"    ✗ Error: Isolation Forest model not found")
        print("    → Run train_model.py first")
        sys.exit(1)
    
    return df


def create_train_test_split(df):
    """Create train/test split."""
    print("\n[3] Creating train/test split...")
    
    # Prepare features and labels
    X = df[['Value_Standardized']].values
    y = df['Label'].values
    
    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=RANDOM_STATE,
        stratify=y  # Maintain class distribution
    )
    
    print(f"    ✓ Train set: {len(X_train):,} samples")
    print(f"      - Normal:  {np.sum(y_train == 0):,}")
    print(f"      - Anomaly: {np.sum(y_train == 1):,}")
    print(f"    ✓ Test set:  {len(X_test):,} samples")
    print(f"      - Normal:  {np.sum(y_test == 0):,}")
    print(f"      - Anomaly: {np.sum(y_test == 1):,}")
    
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate Logistic Regression model."""
    print("\n" + "="*70)
    print("MODEL 1: LOGISTIC REGRESSION")
    print("="*70)
    
    print("\n[4.1] Training Logistic Regression...")
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        solver='lbfgs'
    )
    
    model.fit(X_train, y_train)
    print("    ✓ Training complete")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return model, y_pred, accuracy


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest Classifier."""
    print("\n" + "="*70)
    print("MODEL 2: RANDOM FOREST CLASSIFIER")
    print("="*70)
    
    print("\n[4.2] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1  # Use all available cores
    )
    
    model.fit(X_train, y_train)
    print("    ✓ Training complete")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return model, y_pred, accuracy


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train and evaluate Gradient Boosting Classifier."""
    print("\n" + "="*70)
    print("MODEL 3: GRADIENT BOOSTING CLASSIFIER")
    print("="*70)
    
    print("\n[4.3] Training Gradient Boosting...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    print("    ✓ Training complete")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return model, y_pred, accuracy


def evaluate_model(model_name, y_test, y_pred):
    """Evaluate a single model with detailed metrics."""
    print(f"\n{'-'*70}")
    print(f"Detailed Evaluation: {model_name}")
    print(f"{'-'*70}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted")
    print(f"                Normal  Anomaly")
    print(f"Actual Normal   {cm[0,0]:6d}  {cm[0,1]:7d}")
    print(f"       Anomaly  {cm[1,0]:6d}  {cm[1,1]:7d}")
    
    return cm


def print_comparison_table(results):
    """Print comparison table of all models."""
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    
    print("\n{:<30} {:>12} {:>12} {:>12}".format(
        "Model", "Accuracy", "Precision", "Recall"
    ))
    print("-" * 70)
    
    best_model = None
    best_accuracy = 0
    
    for model_name, metrics in results.items():
        print("{:<30} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            model_name,
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall']
        ))
        
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_model = model_name
    
    print("="*70)
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    return best_model


def save_models(lr_model, rf_model, gb_model):
    """Save trained models."""
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    print("\n[5] Saving model artifacts...")
    
    # Ensure directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save models
    models_to_save = {
        'logistic_regression_model.pkl': lr_model,
        'random_forest_model.pkl': rf_model,
        'gradient_boosting_model.pkl': gb_model
    }
    
    for filename, model in models_to_save.items():
        filepath = os.path.join(MODELS_DIR, filename)
        joblib.dump(model, filepath)
        file_size = os.path.getsize(filepath) / 1024  # KB
        print(f"    ✓ {filename:35s} ({file_size:6.2f} KB)")


def main():
    """Main execution function."""
    
    # Step 1-2: Load data and generate labels
    df = load_data_with_labels()
    
    # Step 3: Create train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(df)
    
    # Step 4: Train models
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    # Model 1: Logistic Regression
    lr_model, lr_pred, lr_acc = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    # Model 2: Random Forest
    rf_model, rf_pred, rf_acc = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Model 3: Gradient Boosting
    gb_model, gb_pred, gb_acc = train_gradient_boosting(X_train, y_train, X_test, y_test)
    
    # Detailed evaluation
    print("\n" + "="*70)
    print("DETAILED MODEL EVALUATION")
    print("="*70)
    
    lr_cm = evaluate_model("Logistic Regression", y_test, lr_pred)
    rf_cm = evaluate_model("Random Forest Classifier", y_test, rf_pred)
    gb_cm = evaluate_model("Gradient Boosting Classifier", y_test, gb_pred)
    
    # Calculate precision and recall for comparison
    def calc_metrics(cm):
        """Calculate precision and recall from confusion matrix."""
        # cm[1,1] = True Positives (anomalies correctly identified)
        # cm[0,1] = False Positives (normal incorrectly labeled as anomaly)
        # cm[1,0] = False Negatives (anomalies incorrectly labeled as normal)
        
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        return precision, recall
    
    lr_prec, lr_rec = calc_metrics(lr_cm)
    rf_prec, rf_rec = calc_metrics(rf_cm)
    gb_prec, gb_rec = calc_metrics(gb_cm)
    
    # Comparison table
    results = {
        'Logistic Regression': {
            'accuracy': lr_acc,
            'precision': lr_prec,
            'recall': lr_rec
        },
        'Random Forest Classifier': {
            'accuracy': rf_acc,
            'precision': rf_prec,
            'recall': rf_rec
        },
        'Gradient Boosting Classifier': {
            'accuracy': gb_acc,
            'precision': gb_prec,
            'recall': gb_rec
        }
    }
    
    best_model = print_comparison_table(results)
    
    # Save models
    save_models(lr_model, rf_model, gb_model)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\n✓ All three supervised models trained successfully")
    print("✓ Models saved to:", MODELS_DIR)
    print(f"\nBest performing model: {best_model}")
    print("\nNext Steps:")
    print("  → Use these models for real-time fault detection")
    print("  → Compare with Isolation Forest (unsupervised) approach")
    print("="*70)


if __name__ == "__main__":
    main()
