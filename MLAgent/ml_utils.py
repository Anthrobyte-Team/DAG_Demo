from flaml import AutoML

def flaml_regression_fit(X, y, time_budget_s=60, metric="mae", log_file="flaml.log"):
    automl = AutoML()
    settings = {
        "time_budget": int(time_budget_s),
        "task": "regression",
        "metric": metric,
        "log_file_name": log_file,
    }
    automl.fit(X_train=X, y_train=y, **settings)
    return automl
