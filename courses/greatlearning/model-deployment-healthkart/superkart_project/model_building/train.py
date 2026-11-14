"""Model training with hyperparameter tuning and MLflow experiment tracking."""
import os
import warnings

# Keep the training log clean (also applies to the parallel worker processes)
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SPLIT_DIR = "splits"
MODEL_DIR = "superkart_project/deployment"
MODEL_PATH = f"{MODEL_DIR}/superkart_sales_model.joblib"
METRICS_PATH = f"{MODEL_DIR}/model_metrics.json"

# Tracking server: GitHub Actions starts one on localhost:5000; locally fall back to ./mlruns
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("superkart-sales-forecasting")

# ---------------------------------------------------------------- data
Xtrain = pd.read_csv(f"{SPLIT_DIR}/Xtrain.csv")
Xtest = pd.read_csv(f"{SPLIT_DIR}/Xtest.csv")
ytrain = pd.read_csv(f"{SPLIT_DIR}/ytrain.csv").squeeze("columns")
ytest = pd.read_csv(f"{SPLIT_DIR}/ytest.csv").squeeze("columns")

numeric_features = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"]
categorical_features = ["Product_Sugar_Content", "Product_Type", "Store_Id",
                        "Store_Size", "Store_Location_City_Type", "Store_Type"]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# ---------------------------------------------------------------- models & grids
candidates = {
    "RandomForest": (
        RandomForestRegressor(random_state=42, n_jobs=1),
        {
            "randomforestregressor__n_estimators": [100, 200],
            "randomforestregressor__max_depth": [10, 20],
            "randomforestregressor__min_samples_leaf": [1, 4],
        },
    ),
    "XGBoost": (
        xgb.XGBRegressor(random_state=42, n_jobs=1, objective="reg:squarederror"),
        {
            "xgbregressor__n_estimators": [100, 200],
            "xgbregressor__max_depth": [3, 5],
            "xgbregressor__learning_rate": [0.05, 0.1],
            "xgbregressor__subsample": [0.8, 1.0],
            "xgbregressor__colsample_bytree": [0.8, 1.0],
        },
    ),
}


def regression_metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
        f"{prefix}_mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


# GridSearchCV parallelises across folds/combinations, so each model uses a single core
cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, (estimator, grid) in candidates.items():
    pipeline = make_pipeline(preprocessor, estimator)
    search = GridSearchCV(
        pipeline, grid, cv=cv, n_jobs=-1,
        scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"}, refit="rmse",
    )

    with mlflow.start_run(run_name=f"{name}-grid-search"):
        search.fit(Xtrain, ytrain)

        # Log every parameter combination tried as a nested run
        cvr = search.cv_results_
        for i, params in enumerate(cvr["params"]):
            with mlflow.start_run(run_name=f"{name}-combo-{i:02d}", nested=True):
                mlflow.log_params({k.split("__")[1]: v for k, v in params.items()})
                mlflow.log_metric("cv_rmse", -cvr["mean_test_rmse"][i])
                mlflow.log_metric("cv_rmse_std", cvr["std_test_rmse"][i])
                mlflow.log_metric("cv_r2", cvr["mean_test_r2"][i])

        # Best model of this family: log best params + train/test metrics
        best = search.best_estimator_
        metrics = {"cv_rmse": float(-search.best_score_)}
        metrics.update(regression_metrics(ytrain, best.predict(Xtrain), "train"))
        metrics.update(regression_metrics(ytest, best.predict(Xtest), "test"))
        mlflow.log_param("model_family", name)
        mlflow.log_params({k.split("__")[1]: v for k, v in search.best_params_.items()})
        mlflow.log_metrics(metrics)

    results[name] = {"model": best, "params": search.best_params_, "metrics": metrics}
    print(f"\n=== {name} ({len(cvr['params'])} combinations x 5 folds) ===")
    print("Best params:", {k.split("__")[1]: v for k, v in search.best_params_.items()})
    print(pd.Series(metrics).round(4).to_string())

# ---------------------------------------------------------------- select & save
best_name = min(results, key=lambda n: results[n]["metrics"]["cv_rmse"])
best = results[best_name]
print(f"\nSelected model (lowest CV RMSE): {best_name}")

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(best["model"], MODEL_PATH, compress=3)   # compressed to keep the repo light
with open(METRICS_PATH, "w") as f:
    json.dump({"model": best_name,
               "best_params": {k.split("__")[1]: v for k, v in best["params"].items()},
               "metrics": {k: round(v, 4) for k, v in best["metrics"].items()}}, f, indent=2)

# Log the selected model file as an artifact of a final summary run
with mlflow.start_run(run_name=f"selected-{best_name}"):
    mlflow.log_param("selected_model", best_name)
    mlflow.log_metrics(best["metrics"])
    mlflow.log_artifact(MODEL_PATH)
    mlflow.log_artifact(METRICS_PATH)

print(f"Best model saved to {MODEL_PATH}")
