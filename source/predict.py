import os
import joblib
import pandas as pd

# 1. ESTABLISH ABSOLUTE REUSABLE DIRECTORY PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "models", "immo_property_sale_XGBoost_model.pkl"))


def load_prediction_pipeline(export_path: str) -> joblib.load:
    """Loads the pre-trained, unified scikit-learn binary object wrapper from disk."""
    if not os.path.exists(export_path):
        raise FileNotFoundError(
            f"\n[ERROR] Model artifact missing! Checked absolute path:\n"
            f"👉 {export_path}\n"
            f"Please run your training script to generate the model file first."
        )
    return joblib.load(export_path)
  

def generate_unseen_property() -> pd.DataFrame:
    """Generates a structured dictionary containing a new Belgian property profile."""

    new_listing = {

        # Numeric Features (Must be complete numbers)
        "latitude": [50.8503],
        "longitude": [4.3517],
        "bedrooms":4,
        "livable_surface": [200.0],
        "bathrooms":2,
        "toilets":2,
        "has_parking":0,
        
        # Categorical Features
        "category": ["apartment"],
        "province": ["Brussels"],
        "epc": ["E"],
        "building_state": ["good"]
    }
    return pd.DataFrame(new_listing)


def main():
    """Execution orchestrator wrapper for streaming property price predictions."""
    # 1. Load your exported training asset
    pipeline = load_prediction_pipeline(MODEL_PATH)
    print(f"Successfully loaded model checkpoint from: '{MODEL_PATH}'")
    
    # 2. Acquire sample live feature dictionary data
    new_data = generate_unseen_property()
    print("\n--- Input Property Features ---")
    print(new_data.to_string(index=False))
    
    # 3. Direct calculation inference
    # [0] extracts the single scalar currency integer from the resulting matrix array
    predicted_price = pipeline.predict(new_data)[0]
    
    # 4. Output validation logs
    print("\n" + "=" * 40)
    print("         PRICE PREDICTION RESULT        ")
    print("=" * 40)
    print(f"Estimated Market Value: {predicted_price:,.2f} EUR")
    print("=" * 40)


if __name__ == "__main__":
    main()
