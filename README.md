# House Price Prediction — Kaggle Ames Dataset

**Notebook-first ML project • EDA • Feature engineering • Regression • Gradio**

This project follows the development organization of the supplied L&T Fake News Classification GitHub example: notebooks for exploration/training, reusable Python modules, saved model artifacts, a separate user interface, dependency file, and run instructions. The algorithms are adapted to house-price regression rather than fake-news classification or RAG.

## 1. Dataset files included

The `data/` folder contains the exact files uploaded for this project:
- `train.csv` — 1,460 rows × 81 columns, including the target `SalePrice`.
- `test.csv` — 1,459 rows × 80 columns, without `SalePrice`.
- `sample_submission.csv` — 1,459 rows with the expected output columns `Id` and `SalePrice`. Its prices are placeholders.

**Do not train on `test.csv` or use `sample_submission.csv` as labels.** Model accuracy is measured using an internal validation split from `train.csv`. After model selection, the selected model is retrained on all labelled training rows and predicts `test.csv` to create `submission.csv`.

## 2. Project structure

```text
House_Price_Prediction_Kaggle_RealData/
|
|-- data/
|   |-- train.csv
|   |-- test.csv
|   |-- sample_submission.csv
|   `-- DATA_DICTIONARY_NOTES.md
|
|-- notebooks/
|   |-- 01_eda.ipynb
|   `-- 02_model_training.ipynb
|
|-- src/
|   |-- data.py
|   |-- eda.py
|   |-- modeling.py
|   `-- predictor.py
|
|-- app/
|   `-- app.py
|
|-- artifacts/
|   |-- best_house_price_model.pkl
|   |-- model_comparison.csv
|   `-- summary.json
|
|-- outputs/
|   `-- charts/
|
|-- train_model.py
|-- generate_report.py
|-- requirements.txt
|-- README.md
|-- submission.csv
`-- House_Price_Project_Report.pdf
```

## 3. Machine-learning workflow

1. Load and inspect the real Kaggle training data.
2. Understand numerical and categorical variables.
3. Handle missing values using the data dictionary: missing categories can mean that a feature is absent (for example, no pool or no garage).
4. Explore distributions, correlations, neighborhood prices, and IQR outliers.
5. Engineer `HouseAge`, `RemodAge`, `GarageAge`, `TotalArea`, `TotalBathrooms`, `TotalPorchSF`, and feature-presence flags.
6. Split labelled data into training and validation sets.
7. Use a preprocessing pipeline: median imputation + standardization for numeric data; categorical missing-value handling + one-hot encoding for categorical data.
8. Compare Linear Regression, Random Forest, and XGBoost. Tune the tree models with randomized search and cross-validation.
9. Evaluate using MAE, RMSE, and R².
10. Refit the selected model on all `train.csv` rows and create `submission.csv` for Kaggle `test.csv`.
11. Load the saved pipeline in a Gradio interface.

The target is log-transformed with `log1p` during fitting to reduce the effect of the right-skewed price distribution. The estimator converts predictions back to USD automatically.

## 4. Feature engineering and leakage prevention

- `HouseAge = YrSold - YearBuilt`
- `RemodAge = YrSold - YearRemodAdd`
- `GarageAge = YrSold - GarageYrBlt` (zero where there is no garage)
- `TotalArea = GrLivArea + TotalBsmtSF`
- `TotalBathrooms = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`
- `TotalPorchSF` is the sum of deck and porch areas.
- `PricePerSqFt = SalePrice / GrLivArea` is descriptive-only. It is excluded from the model because it uses the answer (`SalePrice`) and would cause target leakage.

`Id` is also excluded from model features because it is an identifier, not a meaningful property characteristic.

## 5. Setup on Windows

Open PowerShell in the extracted project root:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Run the notebooks

```powershell
jupyter notebook
```

Run in this order:
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_model_training.ipynb`

The first notebook explains the dataset and produces EDA visualizations. The second trains/evaluates the models and creates the saved artifact and Kaggle submission file.

## 7. Run from the terminal instead

From the project root:

```powershell
python train_model.py
python generate_report.py
python app/app.py
```

The Gradio app normally opens at `http://127.0.0.1:7860`.

## 8. Example results from the included training run

The current run selected **XGBoost** using the lowest average 3-fold cross-validation RMSE (about **$28,458**). On the separate 20% validation holdout, XGBoost achieved MAE about **$15,896**, RMSE about **$27,455**, and R² about **0.902**. Linear Regression had a lower RMSE on this particular holdout (about **$24,294**), despite a much higher average CV RMSE. This illustrates why the project reports both cross-validation and holdout metrics rather than hiding differences between data splits. Rerunning with a different library version or data split can change the results.

## 9. Deliverables

- Two Jupyter notebooks
- Actual Kaggle training/test/sample-submission data
- Saved model pipeline: `artifacts/best_house_price_model.pkl`
- Model comparison: `artifacts/model_comparison.csv`
- Prediction file: `submission.csv`
- EDA and evaluation charts: `outputs/charts/`
- Gradio interface: `app/app.py`
- PDF report: `House_Price_Project_Report.pdf`

## 10. Important limitations

The Ames dataset describes historical home sales in Ames, Iowa, primarily from 2006–2010. Prices are in historical USD. This model is not trained on Nagpur or Indian property data and must not be presented as a current valuation for India. The Gradio form exposes key features and fills other inputs with representative values from the training data, so its predictions are illustrative. The model is an educational estimate, not a certified appraisal.

## 11. Reference implementation

The supplied L&T Fake News Classification repository is used as a project-organization reference: notebooks, reusable source modules, saved model artifacts, app code, dependencies, and documentation. Its NLP/RAG-specific methods are not copied because house-price prediction is a tabular regression problem.

## 🚀 Live Demo

**Try the application here:**

[🏠 House Price Prediction – Live Application](https://l-and-t-assignment-house-price-prediction.onrender.com/)
