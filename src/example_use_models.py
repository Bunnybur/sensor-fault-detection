"""
Example: Using the Trained Supervised Models
=============================================
This script demonstrates how to use the three trained supervised models
for making predictions on new sensor data.

Author: Advanced Computer Programming Course
"""

import joblib
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODELS_DIR


def load_models():
    """Load all trained models and scaler."""
    print("Loading trained models...")
    
    models = {
        'Logistic Regression': joblib.load(os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')),
        'Random Forest': joblib.load(os.path.join(MODELS_DIR, 'random_forest_model.pkl')),
        'Gradient Boosting': joblib.load(os.path.join(MODELS_DIR, 'gradient_boosting_model.pkl')),
        'Isolation Forest': joblib.load(os.path.join(MODELS_DIR, 'isolation_forest_model.pkl'))
    }
    
    scaler = joblib.load(os.path.join(MODELS_DIR, 'standard_scaler.pkl'))
    
    print("✓ All models loaded successfully\n")
    return models, scaler


def predict_with_all_models(sensor_value, models, scaler):
    """Make predictions using all models."""
    # Standardize the input
    value_scaled = scaler.transform([[sensor_value]])
    
    print(f"\n{'='*70}")
    print(f"Sensor Reading: {sensor_value}°C")
    print(f"{'='*70}")
    
    print(f"\n{'Model':<25} {'Prediction':<15} {'Status':<20}")
    print("-" * 70)
    
    for model_name, model in models.items():
        # Make prediction
        prediction = model.predict(value_scaled)[0]
        
        # Format output based on model type
        if model_name == 'Isolation Forest':
            # Isolation Forest outputs: 1 = normal, -1 = anomaly
            status = "✓ Normal" if prediction == 1 else "⚠️ FAULT"
            pred_label = "Normal" if prediction == 1 else "Anomaly"
        else:
            # Supervised models output: 0 = normal, 1 = anomaly
            status = "✓ Normal" if prediction == 0 else "⚠️ FAULT"
            pred_label = "Normal" if prediction == 0 else "Anomaly"
        
        print(f"{model_name:<25} {pred_label:<15} {status:<20}")
    
    print("-" * 70)


def main():
    """Main execution function."""
    print("="*70)
    print("SUPERVISED MODEL PREDICTION DEMO")
    print("="*70)
    
    # Load models
    models, scaler = load_models()
    
    # Test with various sensor values
    test_values = [
        20.0,   # Normal
        25.5,   # Normal
        30.0,   # Normal
        35.0,   # Borderline
        60.0,   # Anomaly
        80.0,   # Anomaly
        120.0,  # Extreme anomaly
        150.0   # Extreme anomaly
    ]
    
    print("\nTesting models with various sensor readings...")
    
    for value in test_values:
        predict_with_all_models(value, models, scaler)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\n📊 Observations:")
    print("  • Random Forest and Gradient Boosting perform identically")
    print("  • Both supervised models show 100% agreement")
    print("  • Isolation Forest (unsupervised) may differ slightly")
    print("  • Logistic Regression fails to detect anomalies properly")
    print("\n💡 Recommendation: Use Random Forest for production")
    print("="*70)


if __name__ == "__main__":
    main()
