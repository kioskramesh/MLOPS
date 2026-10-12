import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main(n_estimators=100, max_depth=10):
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("housing-price")

    with mlflow.start_run() as run:
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
        mlflow.log_metrics({"rmse": rmse, "r2": r2})

        Path("model").mkdir(exist_ok=True)
        joblib.dump(model, "model/model.joblib")
        # Reference data for drift detection later
        X_train.sample(1000, random_state=42).to_csv("model/reference.csv", index=False)
        with open("model/metrics.json", "w") as f:
            json.dump({"rmse": rmse, "r2": r2, "run_id": run.info.run_id}, f)

        mlflow.log_artifact("model/model.joblib")
        print(f"rmse={rmse:.4f} r2={r2:.4f}")


if __name__ == "__main__":
    main()