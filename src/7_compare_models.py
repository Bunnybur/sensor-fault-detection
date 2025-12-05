import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import STANDARDIZED_DATA_PATH, PROCESSED_DATA_DIR, MODELS_DIR, MODEL_PATH, RANDOM_STATE

def load_test_data():
    print("="*70)
    print("MODEL COMPARISON & FINAL TRAINING")
    print("="*70)
    
    print("\n[1] Loading test dataset...")
    test_path = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')
    try:
        df_test = pd.read_csv(test_path)
        print(f"    ✓ Loaded {len(df_test):,} test records")
        return df_test
    except FileNotFoundError:
        print(f"    ✗ Error: Test data not found at {test_path}")
        print("    → Run 'python src/4_train_supervised_models.py' first")
        sys.exit(1)

def load_full_data():
    print("\n[2] Loading full standardized dataset...")
    try:
        df_full = pd.read_csv(STANDARDIZED_DATA_PATH)
        print(f"    ✓ Loaded {len(df_full):,} records for final training")
        return df_full
    except FileNotFoundError:
        print(f"    ✗ Error: Standardized data not found")
        sys.exit(1)

def evaluate_supervised_model(model_name, model_path, X_test, y_test):
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'type': 'supervised'
        }
    except FileNotFoundError:
        print(f"    ⚠️  Model not found: {model_path}")
        return None

def evaluate_isolation_forest(model_path, X_test, y_test):
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred == -1).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary, zero_division=0)
        recall = recall_score(y_test, y_pred_binary, zero_division=0)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'type': 'isolation_forest'
        }
    except FileNotFoundError:
        print(f"    ⚠️  Model not found: {model_path}")
        return None

def evaluate_autoencoder(model_path, threshold_path, X_test, y_test):
    try:
        model = keras.models.load_model(model_path)
        threshold = joblib.load(threshold_path)
        
        reconstructions = model.predict(X_test, verbose=0)
        reconstruction_errors = np.mean(np.abs(X_test - reconstructions), axis=1)
        y_pred = (reconstruction_errors > threshold).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        return {
            'model': model,
            'threshold': threshold,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'type': 'autoencoder'
        }
    except (FileNotFoundError, OSError) as e:
        print(f"    ⚠️  Autoencoder not found: {e}")
        return None

def compare_all_models(df_test):
    print("\n" + "="*70)
    print("EVALUATING ALL MODELS")
    print("="*70)
    
    X_test = df_test[['Value_Standardized']].values
    y_test = df_test['Label'].values
    
    results = {}
    
    print("\n[3] Evaluating supervised models...")
    supervised_models = {
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'Gradient Boosting': 'gradient_boosting_model.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }
    
    for name, filename in supervised_models.items():
        model_path = os.path.join(MODELS_DIR, filename)
        result = evaluate_supervised_model(name, model_path, X_test, y_test)
        if result:
            results[name] = result
            print(f"    ✓ {name}: F1={result['f1']:.4f}, Acc={result['accuracy']:.4f}")
    
    print("\n[4] Evaluating Isolation Forest...")
    iso_path = os.path.join(MODELS_DIR, 'isolation_forest_model.pkl')
    iso_result = evaluate_isolation_forest(iso_path, X_test, y_test)
    if iso_result:
        results['Isolation Forest'] = iso_result
        print(f"    ✓ Isolation Forest: F1={iso_result['f1']:.4f}, Acc={iso_result['accuracy']:.4f}")
    
    print("\n[5] Evaluating Autoencoder...")
    ae_model_path = os.path.join(MODELS_DIR, 'autoencoder_model.keras')
    ae_threshold_path = os.path.join(MODELS_DIR, 'autoencoder_threshold.pkl')
    ae_result = evaluate_autoencoder(ae_model_path, ae_threshold_path, X_test, y_test)
    if ae_result:
        results['Autoencoder'] = ae_result
        print(f"    ✓ Autoencoder: F1={ae_result['f1']:.4f}, Acc={ae_result['accuracy']:.4f}")
    
    return results

def print_comparison_table(results):
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE (SORTED BY F1 SCORE)")
    print("="*70)
    
    print("\n{:<25} {:>12} {:>12} {:>12} {:>12}".format(
        "Model", "F1 Score", "Accuracy", "Precision", "Recall"
    ))
    print("-" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    for model_name, metrics in sorted_results:
        print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            model_name,
            metrics['f1'],
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall']
        ))
    
    print("="*70)
    best_model_name = sorted_results[0][0]
    best_f1 = sorted_results[0][1]['f1']
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   F1 Score: {best_f1:.4f}")
    print(f"   Accuracy: {sorted_results[0][1]['accuracy']:.4f}")
    
    return best_model_name, sorted_results[0][1]

def retrain_best_model(best_model_name, best_model_info, df_full):
    print("\n" + "="*70)
    print("RETRAINING BEST MODEL ON FULL DATASET")
    print("="*70)
    
    print(f"\n[6] Retraining {best_model_name} on full standardized dataset...")
    
    X_full = df_full[['Value_Standardized']].values
    y_full = (df_full['Value'] > 100).astype(int).values
    
    print(f"    • Training samples: {len(X_full):,}")
    print(f"    • Normal: {np.sum(y_full == 0):,}")
    print(f"    • Anomaly: {np.sum(y_full == 1):,}")
    
    model_type = best_model_info['type']
    
    if model_type == 'supervised':
        model = best_model_info['model']
        model.fit(X_full, y_full)
        print(f"    ✓ {best_model_name} retrained successfully")
        
        joblib.dump(model, MODEL_PATH)
        file_size = os.path.getsize(MODEL_PATH) / 1024
        
        print(f"\n[7] Saving final model...")
        print(f"    ✓ Model saved to: {MODEL_PATH}")
        print(f"    ✓ Size: {file_size:.2f} KB")
        print(f"    → FastAPI will now use: {best_model_name}")
        
    elif model_type == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        
        contamination = np.sum(y_full == 1) / len(y_full)
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_full)
        print(f"    ✓ Isolation Forest retrained successfully")
        
        joblib.dump(model, MODEL_PATH)
        file_size = os.path.getsize(MODEL_PATH) / 1024
        
        print(f"\n[7] Saving final model...")
        print(f"    ✓ Model saved to: {MODEL_PATH}")
        print(f"    ✓ Size: {file_size:.2f} KB")
        print(f"    → FastAPI will now use: Isolation Forest")
        
    elif model_type == 'autoencoder':
        print(f"    ℹ️  Autoencoder already trained on normal data from full dataset")
        print(f"    ℹ️  Using existing autoencoder model")
        print(f"    → Note: FastAPI needs custom loading logic for autoencoder")

def main():
    df_test = load_test_data()
    df_full = load_full_data()
    
    results = compare_all_models(df_test)
    
    if len(results) == 0:
        print("\n✗ No trained models found!")
        print("  → Train models first using:")
        print("     1. python src/4_train_supervised_models.py")
        print("     2. python src/5_train_isolation_forest.py")
        print("     3. python src/6_train_autoencoder.py")
        sys.exit(1)
    
    best_model_name, best_model_info = print_comparison_table(results)
    
    retrain_best_model(best_model_name, best_model_info, df_full)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON COMPLETE")
    print("="*70)
    print(f"\n✓ Evaluated {len(results)} models")
    print(f"✓ Best model: {best_model_name}")
    print(f"✓ Final model trained on {len(df_full):,} records")
    print("\nNext Steps:")
    print("  → Start FastAPI server: python src/main.py")
    print("  → Test prediction endpoint with the best model")
    print("="*70)

if __name__ == "__main__":
    main()
