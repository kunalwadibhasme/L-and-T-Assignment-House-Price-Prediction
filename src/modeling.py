"""Full Ames-dataset regression training, tuning, evaluation, and Kaggle prediction."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBRegressor

from .data import (
    load_training_data, load_competition_test, prepare_features,
    make_default_profile, iqr_outlier_summary
)
from .eda import create_eda_charts

RANDOM_STATE = 42

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def build_estimator(X: pd.DataFrame, regressor):
    """Log-transform the skewed SalePrice target; predict() returns dollars."""
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(X)),
        ("model", regressor),
    ])
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )

def _rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))

def train_and_save(project_root, train_path=None, test_path=None):
    root = Path(project_root)
    train_path = Path(train_path or root / "data" / "train.csv")
    test_path = Path(test_path or root / "data" / "test.csv")
    artifacts = root / "artifacts"
    charts = root / "outputs" / "charts"
    artifacts.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)

    raw_train = load_training_data(train_path)
    raw_test = load_competition_test(test_path)
    if "Id" not in raw_train or "Id" not in raw_test:
        raise ValueError("Both train.csv and test.csv must contain Id.")
    if set(raw_train.columns) - {"SalePrice"} != set(raw_test.columns):
        missing = sorted((set(raw_train.columns) - {"SalePrice"}) - set(raw_test.columns))
        extra = sorted(set(raw_test.columns) - (set(raw_train.columns) - {"SalePrice"}))
        raise ValueError(f"Train/test columns do not match. Missing in test: {missing}; extra: {extra}")

    y = pd.to_numeric(raw_train["SalePrice"], errors="coerce")
    valid = y.notna() & (y > 0)
    raw_train = raw_train.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    X = prepare_features(raw_train)
    X_competition = prepare_features(raw_test)
    if list(X.columns) != list(X_competition.columns):
        raise ValueError("Feature columns differ between train and test after feature engineering.")

    # EDA uses the actual Kaggle training records.
    eda = create_eda_charts(raw_train, charts)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Linear Regression": build_estimator(X_train, LinearRegression()),
        "Random Forest": build_estimator(
            X_train,
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
        ),
        "XGBoost": build_estimator(
            X_train,
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
            )
        ),
    }

    parameter_spaces = {
        "Random Forest": {
            "regressor__model__n_estimators": [100, 160, 220],
            "regressor__model__max_depth": [None, 10, 16],
            "regressor__model__min_samples_leaf": [1, 2],
            "regressor__model__max_features": [0.7, 1.0],
        },
        "XGBoost": {
            "regressor__model__n_estimators": [100, 150, 200],
            "regressor__model__max_depth": [2, 3, 4],
            "regressor__model__learning_rate": [0.03, 0.06],
            "regressor__model__subsample": [0.8, 1.0],
            "regressor__model__colsample_bytree": [0.8, 1.0],
            "regressor__model__reg_lambda": [1, 3, 5],
        },
    }

    trained = {}
    cv_rmse = {}
    best_params = {}
    for name, estimator in models.items():
        print(f"\nTraining {name}...")
        if name == "Linear Regression":
            scores = cross_val_score(
                estimator, X_train, y_train,
                scoring="neg_root_mean_squared_error",
                cv=cv, n_jobs=1
            )
            cv_rmse[name] = float(-scores.mean())
            estimator.fit(X_train, y_train)
            trained[name] = estimator
            best_params[name] = "Baseline model; default parameters"
        else:
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=parameter_spaces[name],
                n_iter=3,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                random_state=RANDOM_STATE,
                n_jobs=1,
                refit=True,
                verbose=0,
            )
            search.fit(X_train, y_train)
            trained[name] = search.best_estimator_
            cv_rmse[name] = float(-search.best_score_)
            best_params[name] = search.best_params_
        print(f"{name} cross-validation RMSE: ${cv_rmse[name]:,.2f}")

    # Select using training-fold CV only; do not select based on the holdout set.
    selected_name = min(cv_rmse, key=cv_rmse.get)
    print(f"\nSelected model: {selected_name}")

    result_rows = []
    predictions_by_model = {}
    for name, model in trained.items():
        pred = np.maximum(0, model.predict(X_valid))
        predictions_by_model[name] = pred
        result_rows.append({
            "Model": name,
            "CV RMSE": cv_rmse[name],
            "Validation MAE": float(mean_absolute_error(y_valid, pred)),
            "Validation RMSE": _rmse(y_valid, pred),
            "Validation R2": float(r2_score(y_valid, pred)),
        })

    results = pd.DataFrame(result_rows).sort_values("CV RMSE")
    results.to_csv(artifacts / "model_comparison.csv", index=False)

    best_valid_pred = predictions_by_model[selected_name]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_valid, best_valid_pred, alpha=.6, s=18)
    low = min(float(y_valid.min()), float(best_valid_pred.min()))
    high = max(float(y_valid.max()), float(best_valid_pred.max()))
    ax.plot([low, high], [low, high], "r--", label="Ideal prediction")
    ax.set(title=f"Actual vs predicted — {selected_name}",
           xlabel="Actual SalePrice (USD)", ylabel="Predicted SalePrice (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts / "06_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    residuals = y_valid - best_valid_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=30)
    ax.set(title="Validation residual distribution",
           xlabel="Actual minus predicted (USD)", ylabel="Number of properties")
    fig.tight_layout()
    fig.savefig(charts / "07_residual_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Refit the selected model on ALL labelled training rows for the Kaggle test predictions.
    final_model = build_estimator(X, {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            random_state=RANDOM_STATE, n_jobs=1,
            **{k.replace("regressor__model__", ""): v
               for k, v in best_params["Random Forest"].items()}
        ) if selected_name == "Random Forest" else RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror", random_state=RANDOM_STATE,
            n_jobs=1, tree_method="hist", verbosity=0,
            **{k.replace("regressor__model__", ""): v
               for k, v in best_params["XGBoost"].items()}
        ) if selected_name == "XGBoost" else XGBRegressor(
            objective="reg:squarederror", random_state=RANDOM_STATE,
            n_jobs=1, tree_method="hist", verbosity=0
        ),
    }[selected_name])
    final_model.fit(X, y)

    competition_predictions = np.maximum(0, final_model.predict(X_competition))
    submission = pd.DataFrame({
        "Id": raw_test["Id"].astype(int),
        "SalePrice": competition_predictions,
    })
    submission_path = root / "submission.csv"
    submission.to_csv(submission_path, index=False)

    neighborhoods = sorted(raw_train["Neighborhood"].dropna().astype(str).unique().tolist())
    default_profile = make_default_profile(raw_train)
    bundle = {
        "model": final_model,
        "feature_columns": list(X.columns),
        "default_profile": default_profile,
        "neighborhoods": neighborhoods,
        "selected_model": selected_name,
        "dataset_type": "kaggle_ames_real",
        "currency": "USD",
        "sale_year_min": int(raw_train["YrSold"].min()),
        "sale_year_max": int(raw_train["YrSold"].max()),
        "year_built_min": int(raw_train["YearBuilt"].min()),
        "year_built_max": int(raw_train["YearBuilt"].max()),
    }
    joblib.dump(bundle, artifacts / "best_house_price_model.pkl")

    numeric = raw_train.select_dtypes(include="number").drop(columns=["Id"], errors="ignore")
    corr = numeric.corr(numeric_only=True)["SalePrice"].drop("SalePrice").dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    neighborhood_summary = (
        raw_train.groupby("Neighborhood")["SalePrice"]
        .agg(mean_price="mean", median_price="median", houses="count")
        .sort_values("mean_price", ascending=False)
    )
    summary = {
        "dataset_type": "kaggle_ames_real",
        "rows_raw": int(len(raw_train)),
        "columns_raw": int(raw_train.shape[1]),
        "test_rows": int(len(raw_test)),
        "feature_count_after_engineering": int(X.shape[1]),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "target_mean": float(y.mean()),
        "target_median": float(y.median()),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "target_skewness": float(y.skew()),
        "selected_model": selected_name,
        "model_results": results.to_dict(orient="records"),
        "best_parameters": best_params,
        "top_correlations": {str(k): float(v) for k, v in corr.head(10).items()},
        "top_neighborhoods": [
            {"neighborhood": str(idx), "mean_price": float(row["mean_price"]),
             "median_price": float(row["median_price"]), "houses": int(row["houses"])}
            for idx, row in neighborhood_summary.head(8).iterrows()
        ],
        "outlier_summaries": {
            col: iqr_outlier_summary(raw_train[col])
            for col in ["SalePrice", "GrLivArea", "LotArea"]
        },
        "missing_columns_top": {
            str(k): int(v) for k, v in raw_train.isna().sum().sort_values(ascending=False).head(15).items()
        },
        "submission_file": str(submission_path.name),
        "submission_rows": int(len(submission)),
    }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nValidation comparison:")
    print(results.round(3).to_string(index=False))
    print(f"\nSaved final model: {artifacts / 'best_house_price_model.pkl'}")
    print(f"Created Kaggle prediction file: {submission_path}")
    return summary
