# Ames House Prices — Data Dictionary Notes

The full official field descriptions and category codes are supplied in Kaggle's `data_description.txt`.

## File roles
- `train.csv`: 1,460 labelled rows, 81 columns. `SalePrice` is the target.
- `test.csv`: 1,459 unlabelled rows, 80 columns. It is only for final predictions.
- `sample_submission.csv`: example output schema (`Id`, `SalePrice`); its prices are placeholders.

## Important field groups
- `MSSubClass`: dwelling-type code. It should be treated as categorical, not as a continuous number.
- `LotArea`, `LotFrontage`: plot measurements.
- `Neighborhood`: Ames neighborhood.
- `OverallQual`, `OverallCond`: ordinal ratings from 1 (very poor) to 10 (very excellent).
- `YearBuilt`, `YearRemodAdd`, `YrSold`: used to derive house and renovation age.
- `GrLivArea`, `TotalBsmtSF`: living and basement area.
- `BedroomAbvGr`, `TotRmsAbvGrd`, `FullBath`, `HalfBath`: room and bathroom information.
- `GarageCars`, `GarageArea`: garage capacity and area.
- `SalePrice`: historical sale price in USD; the prediction target.

## Missing-value interpretation
The data dictionary uses `NA` in some category descriptions to mean that a feature is absent:
- `Alley`: no alley access
- `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`: no basement
- `FireplaceQu`: no fireplace
- `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`: no garage
- `PoolQC`: no pool
- `Fence`: no fence
- `MiscFeature`: no miscellaneous feature
- `MasVnrType`: often no masonry veneer

The project converts missing values in these columns to the category `None`. Other missing categorical values are imputed as `Missing` within the model pipeline. Numeric columns with an absent physical feature (for example, basement/garage measurements) are set to zero where appropriate; other numeric gaps such as `LotFrontage` are imputed with the training median.

## Target leakage warning
`PricePerSqFt = SalePrice / GrLivArea` uses the target `SalePrice`. It may be calculated for descriptive analysis only, but it must never be included as an input feature for predicting `SalePrice`.
