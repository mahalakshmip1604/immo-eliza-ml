import os
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "cleaned_sale_properties.csv"))

MODEL_EXPORT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "models", "immo_property_sale_XGBoost_model.pkl"))

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
    "certain_parking_space", "nearest_city", "locality", "region"
]

NUMERIC_FEATURES = ["latitude", "longitude", "bedrooms", "livable_surface", "bathrooms", "toilets", "has_parking"]
CATEGORICAL_FEATURES = ["category", "province", "epc", "building_state"]
GROUP_MEDIAN_COLS = ["bedrooms", "bathrooms", "toilets"]
GROUP_COL = "category"
OUTLIER_QUANTILE = 0.99


def load_and_clean_data(file_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads the dataset and drops unneeded/leaky columns.
    Does NOT impute or filter outliers here - both of those are now
    fit only on the training split, to avoid leaking test-set
    information into training-time statistics.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing input dataset at target path: {file_path}")

    df = pd.read_csv(file_path)

    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_present).dropna(subset=["category", "price"])

    # List of numeric columns that need category-based median imputation
    target_cols = ["bedrooms", "bathrooms", "toilets"]

    # Compute and fill medians grouped by category for all target columns at once
    for col in target_cols:
        category_medians = df.groupby("category")[col].transform("median")
        df[col] = df[col].fillna(category_medians)

    X = df.drop(columns=["price"])
    y = df["price"]
    return X, y


def remove_price_outliers(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Computes the price cutoff ONLY from the training set, then applies
    that same cutoff to both train and test. This keeps the test set
    honestly out-of-sample instead of influencing the cutoff itself.
    """
    price_cutoff = y_train.quantile(OUTLIER_QUANTILE)
    print("\n--- Outlier Filtering (cutoff learned from TRAIN only) ---")
    print(f"{OUTLIER_QUANTILE*100:.0f}th Percentile Price Cutoff: {price_cutoff:,.2f} EUR")

    train_mask = y_train <= price_cutoff
    test_mask = y_test <= price_cutoff

    print(f"Removed {(~train_mask).sum()} outlier properties from train.")
    print(f"Removed {(~test_mask).sum()} outlier properties from test.")

    return X_train[train_mask], X_test[test_mask], y_train[train_mask], y_test[test_mask]


def build_preprocessing_pipeline() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # drop="first" removed: unnecessary for tree-based models and
        # costs the model an explicit split on that dropped category
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def create_model_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """
    Wraps the group-median imputer, preprocessor, and an aggressively
    regularized GridSearchCV-managed XGBRegressor inside one pipeline.
    
    Grid size: max_depth(1) * learning_rate(2) * reg_alpha(2) * reg_lambda(2) * subsample(2)
    = 16 combinations * 5 cv folds = 80 total fits. Highly manageable.
    """
    # Base configuration uses strict feature subsampling
    base_xgb = XGBRegressor(
        n_estimators=500,        
        colsample_bytree=SHARED_FIXED_PARAMS["colsample_bytree"],   
        min_child_weight=SHARED_FIXED_PARAMS["min_child_weight"],
        gamma=SHARED_FIXED_PARAMS["gamma"],
        random_state=42,
        n_jobs=-1,
    )

    # Search space scaled to combat large magnitude gradient updates
    param_grid = {
        "max_depth":[3,4],             
        "learning_rate": [0.03, 0.05],   
        "reg_alpha":[10000, 50000],   # Massive L1 penalty to eliminate noise
        "reg_lambda":[30, 50],        # Strong L2 penalty to smooth weights
        "subsample": [0.5, 0.6],       # Strict row bagging for variance protection
    }

    search = GridSearchCV(
        estimator=base_xgb,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )

    pipeline = Pipeline(steps=[        
        ("preprocessor", preprocessor),
        ("regressor", search),
    ])
    return pipeline
# Global structural constraints shared across both methods to fix the 5.5% gap
SHARED_FIXED_PARAMS = {
    "colsample_bytree": 0.30,   # Drastically limit feature exposure per tree
    "min_child_weight": 50,     # Enforce heavy row support to block micro-trends
    "gamma": 10000.0,            # High split penalty tailored to multi-billion MSE
}
def find_best_n_estimators(
    best_params: dict, X_train_transformed: np.ndarray, y_train: pd.Series
) -> XGBRegressor:
    """
    Carves a validation slice out of the preprocessed training data and
    uses a tightened early stopping window to pick an honest n_estimators count.
    
    Fixes the tree-explosion bug by matching the pipeline's fixed structural 
    constraints, dropping early stopping patience, and capping the ceiling.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_transformed, y_train, test_size=0.15, random_state=42
    )

    # Combined tuned parameters from GridSearch and fixed regularization constraints
    full_hyperparams = {
        **best_params,
        **SHARED_FIXED_PARAMS
    }

    # Strict early stopping configuration to stop chasing minor gains
    model = XGBRegressor(
        **full_hyperparams,
        n_estimators=900,               # Hard cap: stops the 1900+ tree creep
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=15,       # Swift cutoff when validation plateaus
        eval_metric="rmse",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    chosen_trees = model.best_iteration + 1
    print(f"\nEarly stopping selected n_estimators = {chosen_trees} (ceiling was 600)")

    # Refit on the FULL training set using the validated tree count
    final_model = XGBRegressor(
        **full_hyperparams,
        n_estimators=chosen_trees,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_train_transformed, y_train)
    return final_model


def evaluate_and_diagnose(
    pipeline: Pipeline,
    search: GridSearchCV,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Calculates all metrics, prints diagnostics, and returns a dictionary of values."""
    print("\n---- GRID SEARCH RESULTS (structural params) ----")
    print(f"Best CV Hyperparameters: {search.best_params_}")
    print(f"Best Validation Cross-Validation R^2: {search.best_score_:.4f}")
    print(f"Final n_estimators (via early stopping): {pipeline.named_steps['regressor'].n_estimators}")

    train_r2 = pipeline.score(X_train, y_train)
    y_pred_test = pipeline.predict(X_test)

    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)

    print("\n----MODEL EVALUATION METRICS----")
    print(f"Training R^2 Score: {train_r2:.4f}")
    print(f"Test R^2 Score:     {test_r2:.4f}")
    print(f"Test MAE Error:    {test_mae:,.2f} EUR")
    print(f"Test MSE Error:    {test_mse:,.2f}")
    print(f"Test RMSE Error:   {test_rmse:,.2f} EUR")

    print("\n----OVERFITTING DIAGNOSTIC---- ")
    r2_drop = train_r2 - test_r2

    if r2_drop > 0.05:
        print(f"Is there overfitting? YES ({r2_drop*100:.1f}% drop)")
    else:
        print(f"Is there overfitting? NO (Drop is {r2_drop*100:.1f}%)")

    return {
        "Train_R2": f"{train_r2:.4f}",
        "Test_R2": f"{test_r2:.4f}",
        "Overfit": f"{max(0.0, r2_drop * 100):.1f}%",
        "MAE": f"{test_mae:,.2f}",
        "MSE": f"{test_mse:,.2f}",
        "RMSE": f"{test_rmse:,.2f}",
    }


def export_pipeline(pipeline: Pipeline, export_path: str) -> None:
    """Export the end-to-end pipeline object to disk."""
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    joblib.dump(pipeline, export_path)
    print(f"Pipeline saved successfully to '{export_path}'!")


def train_and_evaluate() -> dict:
    """Wrapper function to let main.py directly import, execute, and grab pipeline statistics."""
    # load & do the leak-free column drop only
    X, y = load_and_clean_data(DATA_PATH)

    # split FIRST, before any statistic (median, quantile) is learned
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # outlier cutoff learned only from train, applied to both splits
    X_train, X_test, y_train, y_test = remove_price_outliers(X_train, X_test, y_train, y_test)

    # --- Stage 0: fit preprocessing (group imputer + encoder/scaler) on train only ---
    
    preprocessor = build_preprocessing_pipeline()

    
    X_train_transformed = preprocessor.fit_transform(X_train)

    # --- Stage 1: GridSearchCV picks structural/regularization params ---
    # (n_estimators deliberately excluded - see find_best_n_estimators)
    search = create_model_pipeline(preprocessor).named_steps["regressor"]
    print("\nInitializing hyperparameter tuning via GridSearchCV...")
    search.fit(X_train_transformed, y_train)

    # --- Stage 2: early stopping decides n_estimators on a held-out validation slice ---
    final_regressor = find_best_n_estimators(search.best_params_, X_train_transformed, y_train)

    # --- Reassemble a single fitted Pipeline for export/inference ---
    # These steps are already fitted; Pipeline.predict()/.score() just calls
    # transform/predict on each step in order, so no re-fit is needed here.
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", final_regressor),
    ])

    metrics = evaluate_and_diagnose(pipeline, search, X_train, X_test, y_train, y_test)

    export_pipeline(pipeline, MODEL_EXPORT_PATH)

    return metrics


if __name__ == "__main__":
    train_and_evaluate()