import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# Step 1: Generate realistic sample data
# ----------------------------
np.random.seed(42)

n_samples = 500

data = pd.DataFrame({
    "weight_kg": np.random.uniform(100, 20000, n_samples),
    "volume_m3": np.random.uniform(1, 80, n_samples),
    "distance_km": np.random.uniform(50, 5000, n_samples),
    "fuel_price_usd": np.random.uniform(0.8, 1.8, n_samples),
    "shipping_lane": np.random.choice(["Asia-Europe", "US-Europe", "Asia-US", "Domestic"], n_samples),
    "season": np.random.choice(["Peak", "Off-Peak"], n_samples),
    "carrier_type": np.random.choice(["Air", "Sea", "Road", "Rail"], n_samples)
})

# Target variable with some realistic cost formula
data["freight_cost"] = (
    0.05 * data["weight_kg"] +
    2.5 * data["volume_m3"] +
    0.8 * data["distance_km"] +
    200 * data["fuel_price_usd"] +
    np.where(data["shipping_lane"] == "Asia-Europe", 500, 0) +
    np.where(data["carrier_type"] == "Air", 3000, 0) +
    np.where(data["carrier_type"] == "Sea", 800, 0) +
    np.where(data["season"] == "Peak", 500, 0) +
    np.random.normal(0, 200, n_samples)
)

# ----------------------------
# Step 2: Preprocessing
# ----------------------------
X = data.drop("freight_cost", axis=1)
y = data["freight_cost"]


print("--- Sample of the generated data (long format): ---")
print(data.head())
print("\n--- Data tail: ---")
print(data.tail())


numeric_features = ["weight_kg", "volume_m3", "distance_km", "fuel_price_usd"]
categorical_features = ["shipping_lane", "season", "carrier_type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Transform features first
X_processed = preprocessor.fit_transform(X)

# ----------------------------
# Step 3: Train FLAML AutoML
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

automl = AutoML()
settings = {
    "time_budget": 60,  # seconds
    "metric": "mae",
    "task": "regression",
    "log_file_name": "automl_transport.log"
}

print("⏳ Training AutoML...")
automl.fit(X_train=X_train, y_train=y_train, **settings)

# ----------------------------
# Step 4: Evaluation
# ----------------------------
y_pred = automl.predict(X_test)

print("\n📊 Model Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
import numpy as np

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

print("R2 Score:", r2_score(y_test, y_pred))

print("\n🤖 Best model found by FLAML:")
print(automl.best_estimator)
print("\n🔧 Best hyperparameters:")
print(automl.best_config)
print("\n📑 Model attributes:")
print(automl.model.estimator)

# ----------------------------
# Step 5: User Input for Prediction
# ----------------------------
user_input = {
    "weight_kg": 5000,
    "volume_m3": 20,
    "distance_km": 1200,
    "fuel_price_usd": 1.2,
    "shipping_lane": "Asia-Europe",
    "season": "Peak",
    "carrier_type": "Sea"
}

user_df = pd.DataFrame([user_input])
user_processed = preprocessor.transform(user_df)
predicted_cost = automl.predict(user_processed)[0]

print("\n💰 Predicted Transportation Cost for user input:", round(predicted_cost, 2))

