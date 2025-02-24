# Telecom Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-green)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-yellow)](https://lightgbm.readthedocs.io/)

## 📋 Project Overview

This project implements an end-to-end machine learning solution for predicting customer churn in the telecommunications industry. Customer churn prediction is critical for telecom companies as it enables them to identify customers at risk of cancellation and implement proactive retention strategies.

The solution includes:
- Data preprocessing and feature engineering
- Exploratory data analysis with visualizations
- Model training and hyperparameter optimization
- Evaluation metrics and performance analysis
- A production-ready prediction API for integration

## 🔍 Dataset

The model is trained on a telecom customer dataset with the following information:
- **Demographics**: gender, senior citizen status, partner, dependents
- **Customer account**: tenure, contract type, payment method
- **Services**: phone, internet, additional services
- **Billing**: monthly charges, total charges, paperless billing
- **Target variable**: Churn (Yes/No)

## 🏗️ Project Structure

```
telecom_churn_prediction/
│
├── data/                          # Data storage
│   ├── telco_churn_data.csv       # Raw dataset
│   └── processed/                 # Preprocessed data
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Feature Engineering
│   └── 03_model.ipynb             # Model Development & Evaluation
│
├── models/                        # Trained models
│   ├── final_model.pkl            # Primary production model
│   ├── final_telco_churn_model.pkl  # Alternative model
│   └── model_metadata.json        # Model performance metrics
│
├── src/                           # Source code
│   ├── __init__.py                # Package initialization
│   ├── data_processing.py         # Data processing functions
│   ├── prediction.py              # Prediction module
│   ├── main.py                    # Main application script
│   └── example_usage.py           # Usage examples
│
├── artefacts/                     # Project artifacts
│   └── sweetviz_report.html       # Automated EDA report
│
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/telecom-churn-prediction.git
   cd telecom-churn-prediction
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Command Line Interface

The project provides a command-line interface for the full prediction pipeline:

```bash
# Run the complete pipeline (preprocessing, training, and prediction)
python src/main.py --mode all

# Run only the preprocessing step
python src/main.py --mode preprocess

# Make predictions using a trained model
python src/main.py --mode predict --input-file path/to/data.csv --output-dir path/to/results
```

### Available Options

- `--mode`: Operation mode (`preprocess`, `train`, `predict`, `all`)
- `--input-file`: Path to the input CSV file
- `--output-dir`: Directory to save results
- `--threshold`: Probability threshold for classifying as churn (default: 0.5)

### Python API

The system can also be used programmatically:

```python
# Import necessary functions
from src.data_processing import clean_data, engineer_features
from src.prediction import predict_churn, predict_single_customer

# Predict for a single customer
customer_data = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    # ... other customer attributes
}
prediction = predict_single_customer(customer_data)
print(f"Churn probability: {prediction['churn_probability']:.4f}")

# Batch prediction
import pandas as pd
df = pd.read_csv('new_customers.csv')
results = predict_churn(df)
```

## 🔬 Methodology

### 1. Data Exploration & Preprocessing

- Comprehensive exploratory data analysis
- Handling missing values and outliers
- Feature engineering and transformation
- Data cleaning and standardization

### 2. Model Development

The project compares several machine learning algorithms:
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Advanced techniques implemented:
- SMOTE for handling class imbalance
- Cross-validation for robust evaluation
- Hyperparameter optimization with Optuna
- Feature importance analysis

### 3. Model Evaluation

Performance metrics tracked:
- ROC-AUC
- Precision, Recall, F1-Score
- Confusion Matrix
- Business-focused metrics (revenue impact, retention cost)

### 4. Deployment

The model is ready for deployment with:
- Serialized model artifacts
- A prediction API for integration
- Preprocessing pipeline for new data
- Detailed reporting capabilities

## 🔧 Advanced Configuration

### Customizing the Pipeline

To modify the data processing pipeline:
1. Edit `src/data_processing.py` to add or remove features
2. Modify preprocessing steps as needed
3. Run the pipeline again

### Using a Different Model

To use a different trained model:
1. Replace `models/final_model.pkl` with your trained model
2. Update `models/model_metadata.json` with the new model's information

## 📊 Results & Findings

The final model achieves high predictive accuracy for customer churn with key insights:
- Contract type is the most important predictor
- Customers with month-to-month contracts are at higher risk
- Tenure and monthly charges show strong correlations with churn
- Additional services like online security and tech support reduce churn rate

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 