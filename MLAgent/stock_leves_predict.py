import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import norm

def generate_inventory_data(num_records=500):
    """
    Generates a DataFrame of synthetic inventory data for various SKUs.
    The reorder point is calculated using a standard formula plus noise.
    """
    np.random.seed(42)
    
    skus = [f'SKU-{1000 + i}' for i in range(50)] # 50 unique products
    
    data = []
    
    for _ in range(num_records):
        sku = np.random.choice(skus)
        
        # Demand characteristics
        avg_daily_demand = np.random.randint(10, 200)
        demand_std_dev = avg_daily_demand * np.random.uniform(0.1, 0.5) # Std dev is a % of avg demand
        
        # Lead time characteristics
        avg_lead_time = np.random.randint(5, 45)
        lead_time_std_dev = avg_lead_time * np.random.uniform(0.05, 0.2)
        
        # Service level target
        service_level_target = np.random.choice([0.90, 0.95, 0.98, 0.99])
        
        # --- Calculate the theoretical Reorder Point (ROP) ---
        # ROP = (Average Demand * Average Lead Time) + Safety Stock
        # Safety Stock = Z-score * sqrt((Avg Lead Time * Demand Std Dev^2) + (Avg Demand^2 * Lead Time Std Dev^2))
        
        z_score = norm.ppf(service_level_target) # Z-score for the service level
        
        safety_stock = z_score * np.sqrt(
            (avg_lead_time * demand_std_dev**2) + 
            (avg_daily_demand**2 * lead_time_std_dev**2)
        )
        
        demand_during_lead_time = avg_daily_demand * avg_lead_time
        
        reorder_point = demand_during_lead_time + safety_stock
        
        # Add some noise to make it a more realistic regression problem
        reorder_point *= np.random.uniform(0.95, 1.05)
        
        data.append({
            'sku': sku,
            'avg_daily_demand': avg_daily_demand,
            'demand_std_dev': round(demand_std_dev, 2),
            'avg_lead_time': avg_lead_time,
            'lead_time_std_dev': round(lead_time_std_dev, 2),
            'service_level_target': service_level_target,
            'reorder_point': int(round(reorder_point)) # This is our target variable
        })
        
    return pd.DataFrame(data)

def preprocess_data(df):
    """
    Prepares the data for machine learning.
    """
    # Convert SKU to a category type, which FLAML handles well
    if 'sku' in df.columns:
        df['sku'] = df['sku'].astype('category')
    return df

# --- 1. Generate and Prepare Data ---
print("📦 Generating synthetic inventory data...")
df = generate_inventory_data(num_records=1000)
df_processed = preprocess_data(df.copy())

print("Sample of processed data:")
print(df_processed.head())
print("\nData Info:")
df_processed.info()

# Define features (X) and the target (y)
X = df_processed.drop('reorder_point', axis=1)
y = df_processed['reorder_point']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Configure and Run FLAML AutoML ---
print("\n🤖 Initializing FLAML AutoML for regression...")
automl = AutoML()

# Settings for the AutoML run
settings = {
    "time_budget": 60,       # Total time in seconds for the search
    "metric": 'mae',         # Optimize for Mean Absolute Error
    "task": 'regression',
    "log_file_name": "flaml_inventory_regression.log",
    "seed": 42,
}

print(f"🚀 Starting model training... (Metric: {settings['metric']}, Time budget: {settings['time_budget']}s)")
automl.fit(X_train=X_train, y_train=y_train, **settings)

# --- 3. Evaluate the Best Model ---
print("\n✅ Training complete!")
print("--------------------------------------------------")
print(f"🏆 Best ML model found: {automl.best_estimator}")
print(f"⚙️ Best configuration: {automl.best_config}")
print(f"💯 Best MAE on validation data: {automl.best_loss:.2f} units")
print("--------------------------------------------------")

# Make predictions on the unseen test set
y_pred = automl.predict(X_test)

# Calculate performance metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n📊 Performance on the unseen test set:")
print(f"   R-squared: {r2:.4f}")
print(f"   Mean Absolute Error (MAE): {mae:.2f} units")
print("   (This means the model's predictions are, on average, off by ~{:.0f} units)".format(mae))
print("--------------------------------------------------")


# --- 4. Predict on New User Input ---
def predict_new_reorder_point():
    """
    Takes user input for a new product and predicts the reorder point.
    """
    print("\n📝 Enter details for a new product to predict its reorder point:")
    
    try:
        # Collect user input with validation
        sku = input("   Enter a new SKU name (e.g., SKU-9999): ")
        avg_daily_demand = float(input("   Enter Average Daily Demand (e.g., 50): "))
        demand_std_dev = float(input("   Enter Demand Standard Deviation (e.g., 15): "))
        avg_lead_time = float(input("   Enter Average Supplier Lead Time in days (e.g., 10): "))
        lead_time_std_dev = float(input("   Enter Lead Time Standard Deviation in days (e.g., 2): "))
        
        while True:
            service_level_target = float(input("   Enter Service Level Target (e.g., 0.95 for 95%): "))
            if 0 < service_level_target < 1:
                break
            print("   Error: Service level must be between 0 and 1 (e.g., 0.95). Please try again.")

        # Create a DataFrame from the input
        input_data = {
            'sku': [sku],
            'avg_daily_demand': [avg_daily_demand],
            'demand_std_dev': [demand_std_dev],
            'avg_lead_time': [avg_lead_time],
            'lead_time_std_dev': [lead_time_std_dev],
            'service_level_target': [service_level_target]
        }
        input_df = pd.DataFrame(input_data)
        
        # Apply the same preprocessing
        processed_input_df = preprocess_data(input_df)
        
        # Ensure columns are in the same order as training data
        processed_input_df = processed_input_df[X_train.columns]
        
        # Make the prediction
        predicted_rop = automl.predict(processed_input_df)
        
        print("\n--------------------------------------------------")
        print(f"🔮 Predicted Reorder Point: {int(round(predicted_rop[0]))} units")
        print("   (When stock for this item falls to this level, you should place a new order.)")
        print("--------------------------------------------------")

    except ValueError:
        print("\n❌ Invalid input. Please enter numerical values for the fields.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

# Main loop for user interaction
while True:
    predict_new_reorder_point()
    again = input("Predict another reorder point? (yes/no): ").lower().strip()
    if again != 'yes':
        break

print("\nExiting program.")

