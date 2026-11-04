# Beginner Study Guide — House Price Prediction

This guide explains the project in simple language and gives you short answers for a project presentation or viva.

## 1. What is Artificial Intelligence (AI)?
Artificial Intelligence is the broad field of making computer systems perform tasks that normally require human intelligence, such as understanding language, recognizing images, or making decisions.

## 2. What is Machine Learning (ML)?
Machine Learning is a part of AI in which a computer learns patterns from examples (data) and uses those patterns to make predictions on new examples. Instead of writing every rule manually, we train an algorithm using historical data.

## 3. What type of ML is this project?
This is **supervised machine learning** because our training data contains both:
- Inputs: house characteristics such as area, quality, rooms, and neighborhood.
- Correct output: the historical `SalePrice`.

The algorithm learns the relationship between inputs and the known sale price.

## 4. What is regression?
Regression is a supervised-learning task used to predict a continuous numerical value. House price is a continuous value, so this is a regression problem. If we predicted categories such as cheap/medium/expensive, that would be classification.

## 5. What are features and target?
- **Features (X):** the columns used as inputs to the model, such as `GrLivArea`, `OverallQual`, and `Neighborhood`.
- **Target (y):** the value the model must predict, `SalePrice`.
- `Id` is removed because it only identifies a row and does not describe the property.

## 6. What are the three CSV files?
| File | Purpose |
|---|---|
| `train.csv` | 1,460 labelled houses with `SalePrice`; used for model training and internal validation |
| `test.csv` | 1,459 houses without `SalePrice`; used only to generate final predictions |
| `sample_submission.csv` | Example output structure; its price values are placeholders |

We cannot calculate accuracy on Kaggle's `test.csv` because its actual SalePrice values are not supplied. We create a validation split from `train.csv` for evaluation.

## 7. What is EDA?
Exploratory Data Analysis means inspecting and visualizing the dataset before model training. It helps us understand distributions, missing values, outliers, relationships, and possible data-quality problems.

Examples:
- Histogram: shows how SalePrice is distributed.
- Scatter plot: shows the relationship between living area and SalePrice.
- Box plot: compares price distributions across quality ratings.
- GroupBy: calculates average/median sale price by neighborhood.
- Correlation: measures linear association between numerical columns.

## 8. What are missing values?
A missing value is a blank or unavailable value. We must understand why it is missing before replacing it.

In this Ames dataset, the data dictionary says many missing categorical values mean that the feature does not exist:
- Missing `PoolQC`: no pool.
- Missing `Alley`: no alley access.
- Missing `BsmtQual`: no basement.
- Missing `GarageType`: no garage.
- Missing `FireplaceQu`: no fireplace.

The project maps these known absence categories to `None`. Some numeric measurements are set to zero when the feature is absent. Other numeric missing values, such as `LotFrontage`, are imputed using the median learned from training data.

## 9. What is mean, median, and mode imputation?
- **Mean:** arithmetic average; can be affected by extreme values.
- **Median:** middle value after sorting; more robust to outliers.
- **Mode:** most frequently occurring category/value.
We use median for numeric columns and a constant category (`Missing`) for remaining categorical gaps inside the pipeline.

## 10. What are outliers? What is IQR?
An outlier is a value unusually far from most observations. IQR means Interquartile Range:
- Q1 = 25th percentile
- Q3 = 75th percentile
- IQR = Q3 − Q1
- Lower bound = Q1 − 1.5 × IQR
- Upper bound = Q3 + 1.5 × IQR

The project flags unusual `SalePrice`, `GrLivArea`, and `LotArea` values. We do not automatically delete them because an unusually expensive or large house can be a real property.

## 11. What is feature engineering?
Feature engineering means creating useful input variables from existing columns.

This project creates:
- `HouseAge = YrSold - YearBuilt`
- `RemodAge = YrSold - YearRemodAdd`
- `GarageAge = YrSold - GarageYrBlt`
- `TotalArea = GrLivArea + TotalBsmtSF`
- `TotalBathrooms = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`
- `TotalPorchSF`: sum of deck and porch areas.
- `HasGarage`, `HasBasement`, `HasFireplace`, `HasPool`: binary indicators.

`PricePerSqFt = SalePrice / GrLivArea` is allowed for descriptive analysis only. It is not a model feature because it uses the target SalePrice. Using it would be **target leakage**.

## 12. Why do we encode categorical variables?
Machine-learning algorithms generally need numeric input. Columns such as `Neighborhood`, `HouseStyle`, and `RoofStyle` contain labels, not quantities. One-Hot Encoding creates binary columns for categories. For example, a neighborhood becomes a set of columns indicating which category applies.

`MSSubClass` contains codes such as 20, 30, and 60. The data dictionary says these are dwelling types, so the project treats this field as categorical rather than as a continuous number.

## 13. Why do we scale numerical features?
Scaling puts numeric variables on comparable ranges. StandardScaler transforms a feature by subtracting its training mean and dividing by its training standard deviation. Scaling is important for many linear/distance-based algorithms. Tree-based models generally do not require it, but the shared preprocessing pipeline keeps the experiment consistent.

## 14. Why split the data?
We split labelled data into:
- Training set: used to fit the model.
- Validation/test holdout: kept aside to estimate performance on unseen examples.

This project uses an 80/20 split with `random_state=42` so the split is reproducible. The Kaggle `test.csv` is a separate unlabelled competition file and is not used to calculate validation metrics.

## 15. What is cross-validation?
Cross-validation divides the training portion into several folds. The model trains on some folds and validates on the remaining fold, repeating the process. We average the validation scores to estimate how consistently the model performs. This project uses 3-fold cross-validation during model selection.

## 16. What is hyperparameter tuning?
Hyperparameters are settings chosen before/during model training, such as the number of trees, tree depth, and learning rate. RandomizedSearchCV tries selected combinations and uses cross-validation to find a good configuration.

## 17. Explain the three algorithms

### Linear Regression
Linear Regression estimates a relationship between input features and the target using coefficients. It is a useful baseline and is relatively easy to interpret. It may struggle when relationships are highly nonlinear or when the input columns are strongly correlated.

### Random Forest Regressor
Random Forest builds many decision trees using different samples/features and averages their predictions. It can learn nonlinear relationships and interactions, but a large forest can take more memory and may be less interpretable.

### XGBoost Regressor
XGBoost is a gradient-boosted tree algorithm. It builds trees sequentially, where later trees try to correct errors made by earlier trees. It is powerful for structured/tabular data, but requires tuning to avoid overfitting.

## 18. Why do we use a Pipeline?
A Pipeline connects preprocessing and model training into one object. During cross-validation, the imputer, scaler, and encoder are fitted only on the training fold. This prevents information from validation folds from leaking into preprocessing. We save the whole pipeline so the app uses exactly the same preprocessing as training.

## 19. Why log-transform SalePrice?
SalePrice has a right-skewed distribution: a relatively small number of expensive houses create a long upper tail. `log1p(SalePrice)` compresses large values and can make the learning problem more stable. `TransformedTargetRegressor` automatically converts predictions back to the original dollar scale.

## 20. Explain the evaluation metrics
- **MAE:** average absolute prediction error in dollars. Lower is better.
- **RMSE:** square root of average squared error. It penalizes large mistakes more strongly. Lower is better.
- **R²:** fraction of target variance explained by the model. Closer to 1 is generally better; it can be negative.

Example: if actual price is $200,000 and prediction is $220,000, absolute error is $20,000.

## 21. What is overfitting and underfitting?
- **Overfitting:** the model learns the training examples too closely and performs poorly on unseen data.
- **Underfitting:** the model is too simple to learn the patterns and performs poorly even on training data.
We use validation and cross-validation to detect generalization problems.

## 22. What is target leakage?
Target leakage happens when information that would not be available at prediction time, or information derived from the answer, is used as an input. `PricePerSqFt` uses `SalePrice`, so including it would leak the answer into the model.

## 23. What is Joblib / Pickle?
Joblib serializes a fitted Python object to a file. We save the complete preprocessing and model pipeline as `.pkl`, then load it in the Gradio application. Only load trusted model files because pickle-based formats can execute code when deserialized.

## 24. What is Gradio?
Gradio is a Python library for building a simple web interface around a machine-learning function. Users enter property details, click a button, and see the model's predicted historical price.

The demo form exposes important features. Other columns are filled with representative values from the training data. This makes the UI easy to use but means it is an educational interface, not a full property appraisal form.

## 25. Common viva questions

**Q1. Why is this a regression problem?**  
Because the output `SalePrice` is a continuous numerical value, not a category.

**Q2. What is the target column?**  
`SalePrice`, the actual historical selling price of each house in the training dataset.

**Q3. Why do train.csv and test.csv have different column counts?**  
`train.csv` includes `SalePrice`; `test.csv` does not. Both also include `Id`, which is excluded from model features.

**Q4. Can we evaluate the model using test.csv?**  
No. The actual SalePrice values are not supplied in test.csv. We evaluate using a holdout split from train.csv.

**Q5. Why not fill every missing value with zero?**  
Because some missing values mean a feature is absent, while others mean the measurement is unknown. We use the data dictionary to choose an appropriate treatment.

**Q6. Why use median instead of mean?**  
Median is less affected by extreme values and is often more robust for skewed property measurements.

**Q7. Why remove Id?**  
It is only an identifier and has no reliable property meaning.

**Q8. Why treat MSSubClass as categorical?**  
Its values are codes for dwelling types, not measurements where 60 is three times 20.

**Q9. Why use One-Hot Encoding?**  
It converts text categories into numeric indicator columns that models can process without imposing a false numerical order.

**Q10. Why use three models?**  
To compare a simple linear baseline with two tree-based algorithms that can learn nonlinear patterns.

**Q11. What is the difference between MAE and RMSE?**  
Both measure prediction error. RMSE squares errors before averaging, so large mistakes have more influence. MAE is easier to interpret as average dollar error.

**Q12. What is R²?**  
It measures how much of the variation in the target is explained by the model relative to predicting the mean. A value closer to 1 is generally better.

**Q13. What is cross-validation?**  
It repeats training and validation across multiple folds to estimate model performance more reliably than one split alone.

**Q14. What is hyperparameter tuning?**  
Trying different model settings, such as tree count and depth, to find a configuration with better cross-validation performance.

**Q15. Why save the full pipeline instead of only the regressor?**  
Because new user inputs need the same missing-value handling, scaling, and category encoding used during training.

**Q16. What is the limitation of this dataset?**  
It contains historical sales from Ames, Iowa, not current Indian property transactions. The model should not be used to quote current Nagpur house prices.

**Q17. What happens after model training?**  
We save the selected model pipeline, predict the unlabelled Kaggle test data to create `submission.csv`, and load the saved model in the Gradio app.

## 26. Suggested 60-second project explanation

"My project is an end-to-end house-price prediction system using supervised machine learning regression. I used the Kaggle Ames housing dataset, which contains 1,460 labelled training records and 1,459 unlabelled test records. First, I performed data understanding and EDA to study missing values, distributions, outliers, correlations, and neighborhood-wise price patterns. Then I cleaned the data using the data dictionary, engineered features such as house age and total area, and encoded categorical columns. I compared Linear Regression, Random Forest, and XGBoost using cross-validation and evaluated the models using MAE, RMSE, and R-squared. I saved the preprocessing and model pipeline using Joblib, created predictions for the Kaggle test file, and developed a Gradio interface for interactive predictions. The dataset is historical Ames, Iowa data, so the result is an educational estimate rather than a current Indian property valuation."
