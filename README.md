# Sensor Fault Detection System (Modular Architecture)

**Advanced Computer Programming - Final Project**

A modular IoT fault detection system using both unsupervised (Isolation Forest) and supervised (Random Forest, Gradient Boosting, Logistic Regression) machine learning to identify anomalies in PT100 temperature sensor data, exposed via a RESTful FastAPI backend.

---

## 📁 Project Structure

```
sensor-fault-detection/
│
├── data/                          # Data directory
│   ├── sensor-fault-detection.csv # Raw dataset (62,629 records)
│   ├── processed/                 # Processed data files
│   │   ├── sensor_data_cleaned.csv
│   │   └── sensor_data_standardized.csv
│   └── models/                    # Trained ML models
│       ├── isolation_forest_model.pkl
│       └── standard_scaler.pkl
│
├── src/                           # Source code
│   ├── config.py                  # Centralized configuration
│   ├── data_clean.py              # Step 1: Data cleaning
│   ├── data_standardization.py   # Step 2: Standardization
│   ├── data_analysis.py           # Step 3: Analysis & visualization
│   ├── train_model.py             # Step 4a: Isolation Forest training
│   ├── train_supervised_models.py # Step 4b: Supervised models training
│   ├── example_use_models.py      # Example: Using trained models
│   └── main.py                    # FastAPI application
│
├── run_pipeline.py                # Execute complete pipeline
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline
```bash
python run_pipeline.py
```

This runs all steps automatically:
1. ✅ Data cleaning (drop SensorId, remove nulls)
2. ✅ Data standardization (StandardScaler)
3. ✅ Data analysis & visualization
4. ✅ Model training (Isolation Forest)

To skip visualizations:
```bash
python run_pipeline.py --skip-viz
```

### 3. Start the API Server
```bash
python src/main.py
```

Access the API at: **http://localhost:8000/docs**

---

## 📊 Data Processing Pipeline

### Step 1: Data Cleaning (`data_clean.py`)
**Purpose**: Prepare raw data for analysis

**Operations**:
- Load raw CSV data (62,629 records)
- Drop `SensorId` column (constant value: 1)
- Identify and remove null values
- Save to `data/processed/sensor_data_cleaned.csv`

**Run**:
```bash
python src/data_clean.py
```

---

### Step 2: Data Standardization (`data_standardization.py`)
**Purpose**: Normalize data for ML training

**Operations**:
- Convert timestamps to datetime objects
- Sort data chronologically
- Apply StandardScaler: `z = (x - mean) / std`
- Save scaler for API inference
- Save to `data/processed/sensor_data_standardized.csv`

**Run**:
```bash
python src/data_standardization.py
```

---

### Step 3: Data Analysis (`data_analysis.py`)
**Purpose**: Explore and visualize data

**Features**:
- Descriptive statistics
- IQR-based outlier detection
- Extreme value identification (>100°C)
- Three-plot visualization:
  1. Complete time series with anomalies
  2. Normal operating range (0-60°C)
  3. Standardized values (z-scores)

**Run**:
```bash
python src/data_analysis.py
```

---

### Step 4a: Unsupervised Model - Isolation Forest (`train_model.py`)
**Purpose**: Train anomaly detection model using unsupervised learning

**Algorithm**: Isolation Forest
- **Contamination**: 8% expected anomaly rate
- **n_estimators**: 100 trees
- **Features**: Standardized temperature values

**Outputs**:
- `data/models/isolation_forest_model.pkl`
- Model evaluation metrics
- Top 10 most anomalous readings

**Run**:
```bash
python src/train_model.py
```

---

### Step 4b: Supervised Models (`train_supervised_models.py`)
**Purpose**: Train supervised classification models for comparison

**Models Trained**:
1. **Logistic Regression** - Linear classification
2. **Random Forest Classifier** - Ensemble decision trees
3. **Gradient Boosting Classifier** - Gradient boosted trees

**Training Approach**:
- Uses Isolation Forest predictions as labels
- 80/20 train-test split with stratification
- Comprehensive evaluation with accuracy, precision, recall

**Outputs**:
- `data/models/logistic_regression_model.pkl`
- `data/models/random_forest_model.pkl`
- `data/models/gradient_boosting_model.pkl`
- Comparison table showing best model

**Performance Results**:
| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Logistic Regression | 92.10% | 0.00% | 0.00% |
| **Random Forest** ⭐ | **100.00%** | **100.00%** | **100.00%** |
| Gradient Boosting | 100.00% | 100.00% | 100.00% |

**Run**:
```bash
python src/train_supervised_models.py
```

**Example Usage**:
```bash
python src/example_use_models.py
```

---

## 🌐 API Endpoints

### Root
```http
GET /
```
Returns API information and status.

---

### Predict Fault
```http
POST /predict
Content-Type: application/json

{
  "value": 25.5
}
```

**Response**:
```json
{
  "value": 25.5,
  "status": "Normal",
  "confidence_score": -0.15
}
```

---

### Create Reading
```http
POST /readings
Content-Type: application/json

{
  "timestamp": "2024-12-04T21:00:00+03:00",
  "value": 22.5
}
```

**Response** (auto-classified):
```json
{
  "id": 1,
  "timestamp": "2024-12-04T21:00:00+03:00",
  "value": 22.5,
  "status": "Normal",
  "confidence_score": -0.12
}
```

---

### Get All Readings
```http
GET /readings
```

### Get Specific Reading
```http
GET /readings/{id}
```

### Update Reading
```http
PUT /readings/{id}
Content-Type: application/json

{
  "value": 30.0
}
```

### Delete Reading
```http
DELETE /readings/{id}
```

### Get Statistics
```http
GET /stats
```

**Response**:
```json
{
  "total_readings": 100,
  "normal_count": 92,
  "fault_count": 8,
  "fault_percentage": 8.0
}
```

---

## 🧪 Testing with PowerShell

```powershell
# Test prediction
Invoke-RestMethod -Uri "http://localhost:8000/predict" `
  -Method POST -ContentType "application/json" `
  -Body '{"value": 25.0}'

# Create a reading
Invoke-RestMethod -Uri "http://localhost:8000/readings" `
  -Method POST -ContentType "application/json" `
  -Body '{"timestamp": "2024-12-04T21:00:00+03:00", "value": 22.5}'

# Get all readings
Invoke-RestMethod -Uri "http://localhost:8000/readings"
```

---

## ⚙️ Configuration

All paths and parameters are centralized in `src/config.py`:

```python
# Data paths
RAW_DATA_PATH = 'data/sensor-fault-detection.csv'
CLEANED_DATA_PATH = 'data/processed/sensor_data_cleaned.csv'
STANDARDIZED_DATA_PATH = 'data/processed/sensor_data_standardized.csv'

# Model paths
MODEL_PATH = 'data/models/isolation_forest_model.pkl'
SCALER_PATH = 'data/models/standard_scaler.pkl'

# Model parameters
CONTAMINATION_RATE = 0.08
RANDOM_STATE = 42
N_ESTIMATORS = 100

# API configuration  
API_HOST = "0.0.0.0"
API_PORT = 8000
```

---

## 📚 Academic References

1. **Liu, F. T., Ting, K. M., & Zhou, Z. H.** (2008). *Isolation Forest.* Proceedings of the 2008 Eighth IEEE International Conference on Data Mining, 413-422. doi:10.1109/ICDM.2008.17

2. **Chandola, V., Banerjee, A., & Kumar, V.** (2009). *Anomaly detection: A survey.* ACM Computing Surveys (CSUR), 41(3), 1-58. doi:10.1145/1541880.1541882

3. **Zhao, Y., Nasrullah, Z., & Li, Z.** (2019). *PyOD: A Python Toolbox for Scalable Outlier Detection.* Journal of Machine Learning Research, 20(96), 1-7.

---

## 🎯 Key Features

✅ **Modular Architecture**: Separate scripts for each pipeline stage  
✅ **Centralized Configuration**: Single source of truth for paths  
✅ **Data Pipeline**: Automated cleaning → standardization → analysis → training  
✅ **Unsupervised ML**: Isolation Forest for anomaly detection  
✅ **Supervised ML**: Logistic Regression, Random Forest, Gradient Boosting  
✅ **Model Comparison**: Performance metrics and comparison table  
✅ **RESTful API**: Complete CRUD operations with FastAPI  
✅ **Production-Ready**: Proper error handling, logging, and documentation

---

## 🛠️ Development

### Run API in Development Mode
```bash
uvicorn src.main:app --reload
```

### Run Individual Pipeline Steps
```bash
python src/data_clean.py
python src/data_standardization.py
python src/data_analysis.py
python src/train_model.py
```

---

## 📈 Model Performance

### Unsupervised Learning (Isolation Forest)
- **Anomaly Detection Rate**: ~8% (matches contamination parameter)
- **Extreme Fault Detection**: 100% for values >100°C
- **Normal Classification**: ~92%
- **Features**: Univariate (temperature values only)

### Supervised Learning Comparison
| Model | Accuracy | Precision | Recall | Best Use Case |
|-------|----------|-----------|--------|---------------|
| Random Forest ⭐ | 100.00% | 100.00% | 100.00% | Production deployment |
| Gradient Boosting | 100.00% | 100.00% | 100.00% | High accuracy required |
| Logistic Regression | 92.10% | 0.00% | 0.00% | Not recommended |

**Recommendation**: Use **Random Forest Classifier** for production (perfect accuracy, smaller model size)

---

## 🎓 Course Information

**Course**: Advanced Computer Programming  
**Project Type**: Final Project  
**Technologies**: Python, FastAPI, Scikit-Learn, Pandas, Matplotlib  
**ML Algorithms**: 
- Unsupervised: Isolation Forest
- Supervised: Logistic Regression, Random Forest, Gradient Boosting  
**Architecture**: Edge Computing + RESTful API

---

**Built with ❤️ for Advanced Computer Programming**
