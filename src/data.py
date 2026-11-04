"""Data loading, semantic missing-value handling, and feature engineering."""
from pathlib import Path
import numpy as np
import pandas as pd

TARGET = "SalePrice"
ID_COLUMN = "Id"

# In the Ames data dictionary, blank values in these columns mean the feature is absent.
NO_FEATURE_CATEGORICAL = [
    "Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType",
    "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence",
    "MiscFeature",
]

# Missing numeric values in these columns commonly mean that the related feature is absent.
ZERO_IF_ABSENT_NUMERIC = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageYrBlt", "GarageCars", "GarageArea",
]

def load_training_data(path):
    """Load Kaggle train.csv. It must contain SalePrice."""
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError("Training CSV must contain the SalePrice target column.")
    return df

def load_competition_test(path):
    """Load Kaggle test.csv. It must not contain SalePrice."""
    df = pd.read_csv(path)
    if TARGET in df.columns:
        raise ValueError("Kaggle test.csv should not contain SalePrice.")
    return df

def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean predictors and create derived fields without using SalePrice."""
    X = raw.drop(columns=[ID_COLUMN, TARGET], errors="ignore").copy()

    # MSSubClass is a dwelling-type code (20, 30, 60...), not a continuous measurement.
    if "MSSubClass" in X.columns:
        X["MSSubClass"] = X["MSSubClass"].astype("Int64").astype("string")

    # Normalize text and use plain object/np.nan so scikit-learn imputers can process it.
    text_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()
    if text_columns:
        X = X.astype({col: object for col in text_columns})
    for col in text_columns:
        cleaned = X[col].astype("string").str.strip().replace("", pd.NA)
        X[col] = cleaned.astype(object).where(cleaned.notna(), np.nan)

    for col in NO_FEATURE_CATEGORICAL:
        if col in X.columns:
            X[col] = X[col].fillna("None")

    for col in ZERO_IF_ABSENT_NUMERIC:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    # Feature engineering: all fields below use property information, never SalePrice.
    if {"YrSold", "YearBuilt"}.issubset(X.columns):
        X["HouseAge"] = (pd.to_numeric(X["YrSold"], errors="coerce") -
                         pd.to_numeric(X["YearBuilt"], errors="coerce")).clip(lower=0)

    if {"YrSold", "YearRemodAdd"}.issubset(X.columns):
        X["RemodAge"] = (pd.to_numeric(X["YrSold"], errors="coerce") -
                         pd.to_numeric(X["YearRemodAdd"], errors="coerce")).clip(lower=0)

    if {"YrSold", "GarageYrBlt"}.issubset(X.columns):
        garage_year = pd.to_numeric(X["GarageYrBlt"], errors="coerce")
        sold_year = pd.to_numeric(X["YrSold"], errors="coerce")
        X["GarageAge"] = (sold_year - garage_year).where(garage_year > 0, 0).clip(lower=0)

    if {"GrLivArea", "TotalBsmtSF"}.issubset(X.columns):
        X["TotalArea"] = (
            pd.to_numeric(X["GrLivArea"], errors="coerce") +
            pd.to_numeric(X["TotalBsmtSF"], errors="coerce")
        )

    bathroom_cols = ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
    if all(c in X.columns for c in bathroom_cols):
        X["TotalBathrooms"] = (
            pd.to_numeric(X["FullBath"], errors="coerce").fillna(0)
            + 0.5 * pd.to_numeric(X["HalfBath"], errors="coerce").fillna(0)
            + pd.to_numeric(X["BsmtFullBath"], errors="coerce").fillna(0)
            + 0.5 * pd.to_numeric(X["BsmtHalfBath"], errors="coerce").fillna(0)
        )

    porch_cols = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    if all(c in X.columns for c in porch_cols):
        X["TotalPorchSF"] = X[porch_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)

    if "GarageCars" in X.columns:
        X["HasGarage"] = (pd.to_numeric(X["GarageCars"], errors="coerce").fillna(0) > 0).astype(int)
    if "TotalBsmtSF" in X.columns:
        X["HasBasement"] = (pd.to_numeric(X["TotalBsmtSF"], errors="coerce").fillna(0) > 0).astype(int)
    if "Fireplaces" in X.columns:
        X["HasFireplace"] = (pd.to_numeric(X["Fireplaces"], errors="coerce").fillna(0) > 0).astype(int)
    if "PoolArea" in X.columns:
        X["HasPool"] = (pd.to_numeric(X["PoolArea"], errors="coerce").fillna(0) > 0).astype(int)

    # Ensure the output has no accidental target or ID columns.
    X = X.drop(columns=[ID_COLUMN, TARGET], errors="ignore")
    return X

def make_default_profile(raw_train: pd.DataFrame) -> dict:
    """Create a typical-property profile for UI fields the user does not enter."""
    X = raw_train.drop(columns=[ID_COLUMN, TARGET], errors="ignore").copy()
    profile = {}
    for col in X.columns:
        if col == "MSSubClass":
            mode = X[col].mode(dropna=True)
            profile[col] = str(int(mode.iloc[0])) if len(mode) else "20"
        elif pd.api.types.is_numeric_dtype(X[col]):
            profile[col] = float(X[col].median()) if X[col].notna().any() else 0.0
        else:
            mode = X[col].mode(dropna=True)
            profile[col] = str(mode.iloc[0]) if len(mode) else "Missing"
    return profile

def iqr_outlier_summary(series: pd.Series) -> dict:
    """Flag potential IQR outliers for investigation; do not automatically delete them."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {
        "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
        "lower_bound": float(lower), "upper_bound": float(upper),
        "flagged_rows": int(((values < lower) | (values > upper)).sum()),
    }
