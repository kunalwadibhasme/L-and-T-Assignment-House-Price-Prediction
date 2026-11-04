"""Exploratory data analysis and chart generation for the Ames dataset."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

def create_eda_charts(train_df: pd.DataFrame, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    created = {}

    # 1. Sale price distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(train_df["SalePrice"], bins=35)
    axes[0].set_title("SalePrice distribution")
    axes[0].set_xlabel("Sale price (USD)")
    axes[1].hist(train_df["SalePrice"].clip(upper=train_df["SalePrice"].quantile(.99)), bins=35)
    axes[1].set_title("SalePrice distribution (up to 99th percentile)")
    axes[1].set_xlabel("Sale price (USD)")
    fig.tight_layout()
    p = out / "01_saleprice_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    created["saleprice_distribution"] = str(p)

    # 2. Area and quality
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(train_df["GrLivArea"], train_df["SalePrice"], alpha=.5, s=16)
    axes[0].set(title="Above-ground living area vs SalePrice", xlabel="GrLivArea (sq ft)", ylabel="SalePrice (USD)")
    quality_order = sorted(train_df["OverallQual"].dropna().unique())
    axes[1].boxplot(
        [train_df.loc[train_df["OverallQual"] == q, "SalePrice"].dropna() for q in quality_order],
        tick_labels=[str(int(q)) for q in quality_order]
    )
    axes[1].set(title="Overall quality vs SalePrice", xlabel="OverallQual (1–10)", ylabel="SalePrice (USD)")
    fig.tight_layout()
    p = out / "02_area_quality_relationships.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    created["area_quality"] = str(p)

    # 3. Neighborhood price comparison
    neighborhood = (train_df.groupby("Neighborhood")["SalePrice"]
                    .agg(mean_price="mean", median_price="median", houses="count")
                    .sort_values("mean_price", ascending=False))
    top = neighborhood.sort_values("mean_price", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    top["mean_price"].plot(kind="bar", ax=ax)
    ax.set(title="Average SalePrice by neighborhood", xlabel="Neighborhood", ylabel="Average SalePrice (USD)")
    ax.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    p = out / "03_neighborhood_average_price.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    created["neighborhood"] = str(p)

    # 4. Missing-value chart
    missing = train_df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].head(20).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    missing.plot(kind="barh", ax=ax)
    ax.set(title="Top 20 columns with missing values", xlabel="Missing row count", ylabel="Column")
    fig.tight_layout()
    p = out / "04_missing_values.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    created["missing_values"] = str(p)

    # 5. Correlation heatmap for top numerical predictors
    numeric = train_df.select_dtypes(include="number").drop(columns=["Id"], errors="ignore")
    corr_to_target = numeric.corr(numeric_only=True)["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False)
    selected = ["SalePrice"] + corr_to_target.head(12).index.tolist()
    corr = numeric[selected].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation heatmap: SalePrice and top numeric features")
    fig.tight_layout()
    p = out / "05_correlation_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    created["correlation"] = str(p)

    # Descriptive-only analysis: PricePerSqFt uses the known historical target
    # and must never be included in the prediction feature matrix.
    house_age = (pd.to_numeric(train_df["YrSold"], errors="coerce") -
                 pd.to_numeric(train_df["YearBuilt"], errors="coerce")).clip(lower=0)
    price_per_sqft = (
        pd.to_numeric(train_df["SalePrice"], errors="coerce") /
        pd.to_numeric(train_df["GrLivArea"], errors="coerce").replace(0, float("nan"))
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(house_age.dropna(), bins=30)
    axes[0].set(title="House age at sale", xlabel="Years", ylabel="Number of houses")
    axes[1].hist(price_per_sqft.dropna(), bins=30)
    axes[1].set(title="Price per square foot (descriptive only)", xlabel="USD per sq ft", ylabel="Number of houses")
    fig.tight_layout()
    p = out / "06_house_age_price_per_sqft.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    created["house_age_price_per_sqft"] = str(p)

    return {
        "charts": created,
        "neighborhood_summary": neighborhood.reset_index().to_dict(orient="records"),
        "top_correlations": corr_to_target.head(12).to_dict(),
    }
