import os
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "cleaned_sale_properties.csv"))

MODEL_EXPORT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "models", "immo_property_DT_model.pkl"))

COLS_TO_DROP = [
    "postal_code", "property_id", "posting_date", "property_type", "transaction_type",
    "new_construction", "price_per_sqm", "cadastral_income", "street",  
    "kitchen_equipment", "build_year", "furnished", "heating_type", "glazing_type", 
    "solar_panels", "air_conditioning", "primary_energy_consumption", "electrical_certificate", 
    "facades", "facade_orientation", "garden", "garden_surface", "garden_orientation", 
    "terrace", "terrace_surface", "balcony", "garages", "garage", "indoor_parking", 
    "outdoor_parking", "cellar", "swimming_pool", "elevator", "running_water", 
    "flooding_area_type", "flood_g_score", "flood_p_score", "availability", "land_surface", 
    "number_of_floors", "apartment_floor", "maintenance_cost", "nearest_city_distance_km", 
    "certain_parking_space", "nearest_city","locality", "region" 
]

NUMERIC_FEATURES = ["latitude", "longitude", "bedrooms", "livable_surface", "bathrooms", "toilets", "has_parking"]
CATEGORICAL_FEATURES = ["category", "province", "epc", "building_state"]


def load_and_clean_data(file_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads dataset from disk, drops unneeded features,
    removes extreme price outliers
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing input dataset at target path: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # cleaning rows
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_present).dropna(subset=['category', 'price'])

    # List of numeric columns that need category-based median imputation
    target_cols = ["bedrooms", "bathrooms", "toilets"]

    # Compute and fill medians grouped by category for all target columns at once
    for col in target_cols:
        category_medians = df.groupby("category")[col].transform("median")
        df[col] = df[col].fillna(category_medians)
    
    # OUTLIER REMOVAL: Drop the top 1% most expensive properties(part of Hyper Tuning)
    price_cutoff = df["price"].quantile(0.99)
    print(f"\n--- Outlier Filtering ---")
    print(f"99th Percentile Price Cutoff: {price_cutoff:,.2f} EUR")
    
    initial_count = len(df)
    df = df[df["price"] <= price_cutoff]
    final_count = len(df)
    print(f"Removed {initial_count - final_count} luxury outlier properties.")
    
    # assign features and target
    X = df.drop(columns=["price"])
    y = df["price"]
    return X, y


def build_preprocessing_pipeline() -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="infrequent_if_exist", min_frequency=5), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def create_model_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """ Wraps the preprocessor and a GridSearchCV-managed DecisionTreeRegressor inside a pipeline. """
    
    # 1. Base estimator
    base_tree = DecisionTreeRegressor(random_state=42)
    
    # 2. Parameter grid tailored to optimize your constraints
    param_grid = {
        'max_depth': [6, 7, 8, 9,None],
        'min_samples_split':[60,80,100],
        'min_samples_leaf':[30,40,100],
        'max_features': [0.5, 0.6, 0.7]
    }
    
    # 3. Create the GridSearch estimator
    grid_search_regressor = GridSearchCV(
        estimator=base_tree,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    # 4. Return unified final pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor), 
            ("regressor", grid_search_regressor)  
        ]
    )
    return pipeline


def evaluate_and_diagnose(pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    """Calculates all metrics, prints insights, and returns metrics payloads as a dictionary."""
    
    # Extract grid search results from pipeline step
    grid_search = pipeline.named_steps["regressor"]
    print("\n---- GRID SEARCH RESULTS ----")
    print(f"Best CV Hyperparameters: {grid_search.best_params_}")
    print(f"Best Validation Cross-Validation R²: {grid_search.best_score_:.4f}")

    train_r2 = pipeline.score(X_train, y_train)
    y_pred_test = pipeline.predict(X_test)
    
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)

    print("\n----MODEL EVALUATION METRICS----")    
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Test R² Score:     {test_r2:.4f}")
    print(f"Test MAE Error:    {test_mae:,.2f} EUR")
    print(f"Test MSE Error:    {test_mse:,.2f}")
    print(f"Test RMSE Error:   {test_rmse:,.2f} EUR")
   
    print("\n----OVERFITTING DIAGNOSTIC---- ")    
    r2_drop = train_r2 - test_r2

    if r2_drop > 0.05:
        print(f"Is there overfitting? YES ({r2_drop*100:.1f}% drop)")
    else:
        print(f"Is there overfitting? NO (Drop is {r2_drop*100:.1f}%)")
        
    # Standardised metrics dictionary returned to main.py
    return {
        "Train_R2": f"{train_r2:.4f}",
        "Test_R2": f"{test_r2:.4f}",
        "Overfit": f"{max(0.0, r2_drop * 100):.1f}%",
        "MAE": f"{test_mae:,.2f}",
        "MSE": f"{test_mse:,.2f}",
        "RMSE": f"{test_rmse:,.2f}"
    }


def export_pipeline(pipeline: Pipeline, export_path: str) -> None:
    """export end-to-end pipeline object to disk."""
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    joblib.dump(pipeline, export_path)
    print(f"Pipeline saved successfully to '{export_path}'!")


def train_and_evaluate() -> dict:
    """Wrapper function to let main.py call this script directly and fetch metric stats."""
    # clean & split data
    X, y = load_and_clean_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # preprocessor
    preprocessor = build_preprocessing_pipeline()

    # model creation (now embeds Grid Search)
    pipeline = create_model_pipeline(preprocessor)    

    # model Training (Runs cross-validation grid search across all cores)
    print("\nInitializing parameter tuning via GridSearchCV...")
    pipeline.fit(X_train, y_train)
    
    # model evaluation (now returning metrics payload dict)
    metrics = evaluate_and_diagnose(pipeline, X_train, X_test, y_train, y_test)

    # model saving
    export_pipeline(pipeline, MODEL_EXPORT_PATH)
    
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
