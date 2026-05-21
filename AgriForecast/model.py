import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    features = df[[
        "tanaman",
        "umur_hari",
        "tinggi_cm",
        "jumlah_daun",
        "kelembaban",
        "suhu",
        "curah_hujan",
        "ph_tanah",
    ]]
    X = pd.get_dummies(features, drop_first=True)
    y = df["hasil_panen_kg"]
    return X, y


def simple_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    average = np.mean(np.abs(y_true))
    if average == 0:
        return 0.0
    score = 100.0 * (1 - (mean_absolute_error(y_true, y_pred) / average))
    return max(0.0, min(100.0, score))


def build_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    accuracy = simple_accuracy_score(y_true, y_pred)
    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3),
        "Accuracy": round(accuracy, 2),
    }


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {"MAE": [], "RMSE": [], "R2": []}

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        cloned = clone(model)
        cloned.fit(X_train, y_train)
        preds = cloned.predict(X_val)
        results["MAE"].append(mean_absolute_error(y_val, preds))
        results["RMSE"].append(np.sqrt(mean_squared_error(y_val, preds)))
        results["R2"].append(r2_score(y_val, preds))

    return {
        "MAE": f"{np.mean(results['MAE']):.3f} ± {np.std(results['MAE']):.3f}",
        "RMSE": f"{np.mean(results['RMSE']):.3f} ± {np.std(results['RMSE']):.3f}",
        "R2": f"{np.mean(results['R2']):.3f} ± {np.std(results['R2']):.3f}",
    }


def train_models(df: pd.DataFrame) -> dict:
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(n_estimators=140, random_state=42)
    rf_model.fit(X_train, y_train)

    predictions_lr = linear_model.predict(X_test)
    predictions_rf = rf_model.predict(X_test)

    metrics = {
        "Linear Regression": build_metrics(y_test, predictions_lr),
        "Random Forest": build_metrics(y_test, predictions_rf),
    }

    cv_results = {
        "Linear Regression": cross_validate_model(linear_model, X, y),
        "Random Forest": cross_validate_model(rf_model, X, y),
    }

    feature_importance = {
        col: round(val, 3)
        for col, val in zip(X.columns.tolist(), rf_model.feature_importances_)
    }

    return {
        "models": {
            "Linear Regression": linear_model,
            "Random Forest": rf_model,
        },
        "feature_columns": X.columns.tolist(),
        "metrics": metrics,
        "cv_results": cv_results,
        "feature_importance": feature_importance,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": {
            "Linear Regression": predictions_lr,
            "Random Forest": predictions_rf,
        },
    }


def evaluate_field_validation(models: dict, feature_columns: list, field_df: pd.DataFrame) -> dict:
    df = field_df.copy()
    if df.empty:
        return {}

    X, y = prepare_features(df)
    for column in feature_columns:
        if column not in X.columns:
            X[column] = 0
    X = X[feature_columns]

    results = {}
    for model_name, model in models.items():
        predictions = model.predict(X)
        results[model_name] = build_metrics(y, predictions)
    return results


def predict_crop_yield(models: dict, feature_columns: list, input_data: dict) -> dict:
    df_input = pd.DataFrame([input_data])
    X_input = pd.get_dummies(df_input)
    for column in feature_columns:
        if column not in X_input.columns:
            X_input[column] = 0
    X_input = X_input[feature_columns]
    results = {
        model_name: float(model.predict(X_input)[0])
        for model_name, model in models.items()
    }
    return results
