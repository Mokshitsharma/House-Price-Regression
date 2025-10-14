# house_price_regression.py
# -------------------------
# Generic house price regressor for ANY tabular CSV you have.
# Expected: one target column named 'price' (you can change TARGET), others are features.
# Handles numeric + categorical columns, missing values, scaling, and tries two models.

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

CSV_PATH = os.environ.get("HOUSE_CSV", "housing.csv")   # put your Kaggle/other CSV here
TARGET = os.environ.get("TARGET", "price")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ----- Load or fabricate tiny demo -----
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    # tiny synthetic sample so the script runs
    rng = np.random.default_rng(42)
    n = 150
    df = pd.DataFrame({
        "area_sqft": rng.integers(500, 3500, n),
        "bedrooms": rng.integers(1, 5, n),
        "bathrooms": rng.integers(1, 4, n),
        "city": rng.choice(["Delhi","Mumbai","Bengaluru","Pune"], n),
        "age_years": rng.integers(0, 30, n),
    })
    # price = 3000*area + 2L per bedroom + noise, scaled
    noise = rng.normal(0, 5e5, n)
    df[TARGET] = df["area_sqft"]*3000 + df["bedrooms"]*2e5 - df["age_years"]*3e4 + noise

print(f"Loaded dataset shape: {df.shape}")
assert TARGET in df.columns, f"Target column '{TARGET}' not found."

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Identify column types
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])

# Two models to compare
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_name, best_pipe, best_rmse = None, None, float("inf")
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    print(f"\n=== {name} ===")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²:   {r2:,.3f}")

    if rmse < best_rmse:
        best_name, best_pipe, best_rmse = name, pipe, rmse

joblib.dump(best_pipe, f"{MODEL_DIR}/house_price_model.joblib")
print(f"\nSaved best model: {best_name} to {MODEL_DIR}/house_price_model.joblib")

# Inference example
sample = X.iloc[[0]].to_dict(orient="records")[0]
print("\nSample input:", sample)
print("Predicted price:", best_pipe.predict(pd.DataFrame([sample]))[0])
