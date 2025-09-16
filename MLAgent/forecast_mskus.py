import pandas as pd
import numpy as np

# --- 1. Generate Realistic Sample Data ---
# Create a date range for our data
dates = pd.date_range(start='2024-01-01', end='2025-07-31', freq='D')
skus = ['SKU_A', 'SKU_B', 'SKU_C']
data = []

# Generate data for each SKU with different patterns
for sku in skus:
    # Base sales with some noise
    sales = np.random.randint(50, 100, size=len(dates))
    
    # Add trend and seasonality to make it realistic
    if sku == 'SKU_A': # SKU with steady growth
        trend = np.linspace(0, 50, len(dates))
        sales = sales + trend
    elif sku == 'SKU_B': # SKU with weekly seasonality (weekends are better)
        seasonality = np.array([5, 0, 0, 0, 10, 40, 50] * (len(dates) // 7 + 1))[:len(dates)]
        sales = sales + seasonality
    else: # SKU_C is more volatile
        sales = np.random.randint(20, 150, size=len(dates))

    # Create a DataFrame for this SKU
    sku_df = pd.DataFrame({'date': dates, 'SKU': sku, 'sales': sales})
    data.append(sku_df)

# Combine all SKU data into a single DataFrame
df = pd.concat(data)

# Add an external feature (exogenous regressor): a promotion flag
# Let's say we run promotions randomly on some days
df['promotion'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
# Make promotions more effective
df['sales'] = df['sales'] * (1 + df['promotion'] * 0.5) 
df['sales'] = df['sales'].astype(int)

print("--- Sample of the generated data (long format): ---")
print(df.head())
print("\n--- Data tail: ---")
print(df.tail())


# --- 2. Split Data into Training and Testing ---
# For time series, the split must be chronological.
# We will train on data up to the end of June 2025 and predict for July 2025.
horizon = 31 # Predict for the 31 days of July
split_date = df['date'].max() - pd.Timedelta(days=horizon-1)

train_df = df[df['date'] < split_date]
test_df = df[df['date'] >= split_date]

# Separate features (X) and the target variable (y)
X_train = train_df.drop(columns='sales')
y_train = train_df['sales']

X_test = test_df.drop(columns='sales')
y_test = test_df['sales']
print("X_test, y_test\n")
print(X_test)
print(y_test)





from flaml import AutoML

# --- 3. Initialize and Configure AutoML for Forecasting ---
automl = AutoML()

# Settings for the AutoML run
settings = {
    "time_budget": 120,  # seconds, time limit for the search
    "metric": "mape",    # Metric to optimize for: Mean Absolute Percentage Error
    "task": "ts_forecast", # Specify the task as time series forecasting
    "log_file_name": "flaml_forecast.log", # Log file
    "eval_method": "holdout", # Use a holdout validation set
    "label": "sales", # The name of the target column
}

# --- 4. Fit the Model ---
# This is the key part for multi-SKU forecasting
automl.fit(
    X_train=X_train,
    y_train=y_train,
    period=horizon,  # The number of periods to forecast
    time_index="date", # The column with the date/time information
    ts_column_name="SKU", # The column identifying each time series (each SKU)
    **settings
)

# --- 5. Review the Results ---
print("\n--- AutoML Run Summary ---")
print("Best ML model found:", automl.best_estimator)
print("Best hyperparameters:", automl.best_config)
print(f"Best MAPE on validation data: {automl.best_loss:.4f}")

# --- 6. Make Future Predictions ---
# FLAML needs the future values of exogenous variables for all SKUs
# Here, we use the 'promotion' column from our test set
future_X = X_test.copy() 

# Use the predict method
predictions = automl.predict(future_X)

# --- 7. Evaluate the Forecast ---
# Create a results DataFrame to compare predictions with actuals
results_df = pd.DataFrame({
    'date': X_test['date'],
    'SKU': X_test['SKU'],
    'actual_sales': y_test,
    'predicted_sales': predictions
})

print("\n--- Forecast Results for July 2025: ---")
print(results_df.head())

# Calculate overall MAPE
from sklearn.metrics import mean_absolute_percentage_error
overall_mape = mean_absolute_percentage_error(results_df['actual_sales'], results_df['predicted_sales'])
print(f"\nOverall MAPE on Test Set: {overall_mape:.4f}")

# Calculate MAPE per SKU
for sku in results_df['SKU'].unique():
    sku_results = results_df[results_df['SKU'] == sku]
    sku_mape = mean_absolute_percentage_error(sku_results['actual_sales'], sku_results['predicted_sales'])
    print(f"MAPE for {sku}: {sku_mape:.4f}")
