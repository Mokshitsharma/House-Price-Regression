# House Price Regression (Generic Tabular Regressor)

A Python script for training, evaluating, and saving a regression model to predict house prices using any tabular CSV data. Handles numeric and categorical features, missing values, scaling, and model selection. Designed for easy use with your own dataset.

## Features

- Loads any housing CSV dataset (customizable path and target column)
- Handles missing values, numeric/categorical preprocessing, and feature encoding automatically
- Trains and compares Linear Regression and Random Forest models
- Evaluates performance using MAE, RMSE, and R² metrics
- Saves the best-performing model for reuse
- Generates synthetic demo data if no CSV exists, so script always runs

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- joblib

Install dependencies:

```bash
pip install pandas numpy scikit-learn joblib
```

## Usage

1. **Place your dataset:**  
   Use any tabular CSV file containing one target column (default: `price`) and feature columns.  
   Sample ways to set CSV path and target:

   ```bash
   export HOUSE_CSV="housing.csv"
   export TARGET="SalePrice"
   python house_price_regression.py
   ```

   Or just run the script and it will generate a synthetic demo if no data is found:

   ```bash
   python house_price_regression.py
   ```

2. **Script output:**  
   - Trains two models (Linear Regression and Random Forest), prints performance, and picks the best.
   - Saves the best trained model as `models/house_price_model.joblib`.
   - Shows a sample prediction on one input row.

## Customization

- `CSV_PATH`: Path to your CSV dataset (default: `housing.csv`, can be set with `HOUSE_CSV` env variable).
- `TARGET`: Name of the target column (default: `price`, can be set with `TARGET` env variable).
- Supports numeric and categorical columns automatically.

## Example Data Schema

| area_sqft | bedrooms | bathrooms | city    | age_years | price     |
|-----------|----------|-----------|---------|-----------|-----------|
| 1200      | 3        | 2         | Mumbai  | 5         | 4,500,000 |
| 2100      | 4        | 3         | Delhi   | 10        | 8,250,000 |

## Output

- A `models/house_price_model.joblib` file containing a trained pipeline (preprocessing + best model).
- Printed evaluation metrics (MAE, RMSE, R²).
- Example prediction using the trained model.

## License

MIT License — free for personal and educational use.

***
