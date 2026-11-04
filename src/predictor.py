"""Prediction helper for the Gradio app."""
from pathlib import Path
import joblib
import pandas as pd
from .data import prepare_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "best_house_price_model.pkl"

def load_model_bundle(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Model not found. Run `python train_model.py` first.")
    # Only load a trusted model file created by this project.
    return joblib.load(path)

def predict_from_inputs(
    living_area, basement_area, bedrooms, total_rooms,
    quality, condition, garage_cars, year_built, year_sold,
    neighborhood, bundle=None
):
    bundle = bundle or load_model_bundle()
    if living_area <= 0 or basement_area < 0:
        raise ValueError("Living area must be positive and basement area cannot be negative.")
    if bedrooms < 0 or total_rooms < 1 or bedrooms > total_rooms:
        raise ValueError("Please check the bedroom and total-room counts.")
    if not 1 <= int(quality) <= 10 or not 1 <= int(condition) <= 10:
        raise ValueError("Quality and condition must be between 1 and 10.")
    if garage_cars < 0:
        raise ValueError("Garage capacity cannot be negative.")
    if year_built < bundle["year_built_min"] or year_built > year_sold:
        raise ValueError(
            f"Year built must be between {bundle['year_built_min']} and the selected sale year."
        )
    if year_sold < bundle["sale_year_min"] or year_sold > bundle["sale_year_max"]:
        raise ValueError(
            f"The Ames training records cover sale years {bundle['sale_year_min']}–{bundle['sale_year_max']}."
        )
    if neighborhood not in bundle["neighborhoods"]:
        raise ValueError("Please select a neighborhood from the training dataset.")

    row = dict(bundle["default_profile"])
    row.update({
        "GrLivArea": float(living_area),
        "TotalBsmtSF": float(basement_area),
        "BedroomAbvGr": int(bedrooms),
        "TotRmsAbvGrd": int(total_rooms),
        "OverallQual": int(quality),
        "OverallCond": int(condition),
        "GarageCars": int(garage_cars),
        "YearBuilt": int(year_built),
        "YrSold": int(year_sold),
        "Neighborhood": neighborhood,
    })
    X = prepare_features(pd.DataFrame([row]))
    X = X.reindex(columns=bundle["feature_columns"])
    price = float(bundle["model"].predict(X)[0])
    return max(0.0, price)
