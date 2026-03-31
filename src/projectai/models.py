"""Model helpers: training and evaluation utilities."""
from typing import Dict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np


def evaluate_model(model, X, y) -> Dict[str, float]:
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, preds)
    return {'MAE': float(mae), 'MSE': float(mse), 'RMSE': float(rmse), 'R2': float(r2)}


def train_linear(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_random_forest(X, y, n_estimators=100, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X, y)
    return model


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
