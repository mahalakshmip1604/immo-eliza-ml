# Immo Eliza ML - Real Estate Price Prediction Model

A machine learning models created for the real estate company 'Immo Eliza' to predict sale prices of real estate properties in Belgium.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results &amp; Performance](#results--performance)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Running the Pipeline](#running-the-pipeline)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

### Objective

Develop a comprehensive machine learning pipeline to predict real estate property prices across Belgium using three different models: Linear Regression, Decision Tree, and XGBoost. The project compares model performance and identifies the best approach for price prediction.

### Business Impact

- Provide accurate and reliable price estimates for properties in Belgium
- Support data-driven decision-making for Immo Eliza
- Enable competitive pricing strategies based on property features
- Identify market patterns and key price drivers

### Models Trained

- **Linear Regression** - Baseline model for interpretability
- **Decision Tree** - Tuned with GridSearchCV for optimal performance
- **XGBoost** - Advanced gradient boosting model with hyperparameter optimization

## ✨ Features

- **Comprehensive Property Analysis**: Analyzes multiple property attributes

  - Location (province, city)
  - Physical dimensions (living area, land area, construction year)
  - Property characteristics (type, condition, rooms, bathrooms)
  - Amenities (parking)
- **Advanced ML Pipeline**:

  - Automated data preprocessing and feature engineering
  - Multi-model training and comparison
  - Hyperparameter tuning using GridSearchCV
  - Cross-validation for robust evaluation
  - Detailed performance metrics and comparison
- **Production-Ready Code**:  Clean, modular Python structure with proper error handling

## 📊 Dataset

### Data Source

Real estate property data from Belgium, pre-processed and cleaned for ML.

### Dataset Files

```
data/
├── cleaned_sale_properties.csv    (3.86 MB - Sale property listings)
└── cleaned_rent_properties.csv    (1.32 MB - Rental property listings)
```

### Data Characteristics

- **Sale Properties**: 3,866,981 bytes of cleaned property sale data
- **Rental Properties**: 1,324,632 bytes of cleaned rental data
- **Geographic Coverage**: Entire Belgium
- **Target Variable**: Property Price (in EUR)

### Key Features

- Living area (m²)
- Land area (m²)
- Number of rooms & bathrooms
- Property type (apartment, house, villa, etc.)
- Location (province)
- EPC
- Property condition and amenities

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git

### Setup Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/mahalakshmip1604/immo-eliza-ml.git
   cd immo-eliza-ml
   ```
2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Verify installation**

   ```bash
   python -c "import pandas as pd; import sklearn; import xgboost; print('✓ All dependencies installed successfully!')"
   ```

## 🚀 Usage

### Quick Start - Run the Complete Pipeline

```bash
python main.py
```

This executes the master pipeline that:

1. Trains Linear Regression model
2. Tunes Decision Tree with GridSearchCV
3. Optimizes XGBoost with GridSearchCV
4. Displays comprehensive performance comparison

### Expected Output

```
============================================================
        STARTING MASTER MACHINE LEARNING PIPELINE  
============================================================

🚀 [1/3] Training Linear Regression...
🚀 [2/3] Tuning Decision Tree (GridSearchCV)...
🚀 [3/3] Optimizing XGBoost (GridSearchCV)...

============================================================================================
                        FINAL MODEL PERFORMANCE SUMMARY                        
============================================================================================
Model Name          | Train R²   | Test R²    | Overfit Drop | Test MAE     | Test MSE         | Test RMSE   
------------------------------------------------------------------------------------------------------------
Linear Regression    | 0.5660     | 0.6158     | 0.0%         | 137,208.52   | 47,353,848,598.38 | 217,609.39  
Decision Tree        | 0.7349     | 0.6964     | 3.9%         | 114,141.29   | 37,423,465,297.41 | 193,451.45  
XGBoost              | 0.8154     | 0.7943     | 2.1%         | 231.78       | 125,104.21        | 353.70           
============================================================================================

Total Pipeline Execution Time: XX.XX seconds
```

### Detailed Usage

#### 1. Training Individual Models

Access the `source/` directory to train specific models:

```python
from source import train_linear_regression_model
results = train_linear_regression_model.train_and_evaluate()
```

```python
from source import train_DT_model
results = train_DT_model.train_and_evaluate()
```

```python
from source import train_XGBoost_model
results = train_XGBoost_model.train_and_evaluate()
```

#### 2. Make Predictions

Load a trained model from the `models/` directory:

```python
import pickle
import pandas as pd

# Load trained model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare property data
property_features = pd.DataFrame({
    'living_area': [120],
    'land_area': [500],
    'rooms': [3],
    'bathrooms': [2],
    # ... other features
})

# Get price prediction
predicted_price = model.predict(property_features)
print(f"Predicted Price: €{predicted_price[0]:,.2f}")
```

## 🧠 Model Architecture

### Pipeline Overview

```
Data Preparation
        ↓
[cleaned_sale_properties.csv]
        ↓
Data Cleaning & Preprocessing
        ↓
Feature Engineering
        ↓
Feature Scaling/Encoding
        ↓
├─→ Linear Regression
├─→ Decision Tree (GridSearchCV)
└─→ XGBoost (GridSearchCV)
        ↓
Model Evaluation & Comparison
        ↓
Best Model Selection
```

### Algorithms & Methods

#### 1. Linear Regression

- **Purpose**: Baseline model for comparison
- **Approach**: Simple linear relationship between features and price
- **Use Case**: Quick estimation, interpretability

#### 2. Decision Tree (GridSearchCV)

- **Purpose**: Capture non-linear patterns
- **Hyperparameter Tuning**: Optimized max_depth, min_samples_split, min_samples_leaf
- **Advantages**: Interpretable decision rules, handles non-linear relationships

#### 3. XGBoost (GridSearchCV)

- **Purpose**: Best-performing model with ensemble learning
- **Hyperparameter Tuning**: Optimized learning rate, max_depth, n_estimators
- **Advantages**: Superior accuracy, handles complex interactions, robust to outliers

### Evaluation Metrics

- **R² Score**: Proportion of variance explained (0-1, higher is better)
- **MAE**: Mean Absolute Error (average absolute prediction error)
- **MSE**: Mean Squared Error (penalizes larger errors)
- **RMSE**: Root Mean Squared Error (in same units as price)
- **Overfit Drop**: Difference between Train R² and Test R² (lower is better)

## 📁 Project Structure

```
immo-eliza-ml/
├── README.md                          # Project documentation
├── main.py                            # Master pipeline orchestrator
├── requirements.txt                   # Python dependencies
│
├── data/                              # Dataset directory
│   ├── cleaned_sale_properties.csv   # Sale property data (3.86 MB)
│   └── cleaned_rent_properties.csv   # Rental property data (1.32 MB)
│
├── source/                            # Source code modules
│   ├── __init__.py
│   ├── train_linear_regression_model.py 
│   ├── train_DT_model.py         
│   ├── train_XGBoost_model.py   
│
├── models/                            # Trained model storage
│   ├── immo_property_regressor_model.pkl
│   ├── immo_property_DT_model.pkl
│   ├── immo_property_XGBoost_model.pkl

│
└── .gitignore                         # Git ignore rules
```

## 📦 Dependencies

### Python Libraries

```
matplotlib==3.11.0           # Data visualization
matplotlib-inline==0.2.2     # Jupyter notebook plotting
numpy==2.5.0                 # Numerical computing
pandas==3.0.3                # Data manipulation
scikit-learn==1.9.0          # Machine learning models
scipy==1.18.0                # Scientific computing
seaborn==0.13.2              # Statistical visualization
xgboost==3.3.0               # Gradient boosting
```

### Installation

Install all dependencies with:

```bash
pip install -r requirements.txt
```

### Python Version

- **Minimum**: Python 3.8
- **Recommended**: Python 3.10+

## ▶️ Running the Pipeline

### Step-by-Step Execution

1. **Ensure all data files are in place**

   ```bash
   ls data/
   # Should show: cleaned_sale_properties.csv, cleaned_rent_properties.csv
   ```
2. **Run the complete pipeline**

   ```bash
   python main.py
   ```
3. **Monitor progress**

   - The pipeline displays progress for each of the 3 models
   - GridSearchCV provides hyperparameter tuning updates
   - Final performance summary is displayed with all metrics
4. **Check results**

   - Review the "FINAL MODEL PERFORMANCE SUMMARY" table
   - Compare R² scores, MAE, RMSE across models
   - Check "Overfit Drop" to ensure models generalize well

### Output Files

Models are saved to the `models/` directory:

- `linear_regression_model.pkl`
- `decision_tree_model.pkl`
- `xgboost_model.pkl`
- Feature scalers and encoders

### Troubleshooting

**Error: "Module not found: source"**

- Ensure you're running from the project root directory
- Verify all files in `source/` directory exist

**Error: "FileNotFoundError: cleaned_sale_properties.csv"**

- Check data files are in `data/` directory
- Verify file names match exactly

**Memory issues with large datasets**

- Reduce batch size in preprocessing
- Use sample of data for testing

## 📈 Results & Performance

The model performance summary includes:

- **Train R²**: Model performance on training data
- **Test R²**: Model generalization on unseen test data
- **Overfit Drop**: Overfitting indicator (Train R² - Test R²)
- **MAE**: Average absolute prediction error in EUR
- **MSE**: Mean squared error metric
- **RMSE**: Root mean squared error in EUR

### Model Comparison Table

The pipeline generates a detailed comparison table showing all metrics side-by-side for easy interpretation.

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork the repository**

   ```bash
   git clone https://github.com/[your-username]/immo-eliza-ml.git
   ```
2. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**

   - Follow PEP 8 style guidelines
   - Add docstrings to functions
   - Include type hints
4. **Test your changes**

   ```bash
   python main.py
   ```
5. **Commit and push**

   ```bash
   git add .
   git commit -m "Add: description of changes"
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request**

   - Describe your changes clearly
   - Reference any related issues

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions focused and modular

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Mahalakshmi P**

- GitHub: [@mahalakshmip1604](https://github.com/mahalakshmip1604)

## 📞 Support

For questions, issues, or suggestions:

- Open a [GitHub Issue](https://github.com/mahalakshmip1604/immo-eliza-ml/issues)
- Check existing documentation
- Review the main.py file for pipeline details

## 🙏 Acknowledgments

- Immo Eliza for the project opportunity
- Python ML community for excellent libraries
- XGBoost, scikit-learn, and pandas teams

---

**Last Updated**: July 2026
**Version**: 1.0.0
**Status**: Production Ready ✓
