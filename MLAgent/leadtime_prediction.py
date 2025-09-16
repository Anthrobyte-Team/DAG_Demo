import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import datetime

def generate_supplier_data(num_records=200):
    """
    Generates a DataFrame of synthetic supplier purchase order data.
    """
    np.random.seed(42)
    
    suppliers = {
        'SUP-101': {'location': 'China', 'base_lead_time': 30},
        'SUP-202': {'location': 'Germany', 'base_lead_time': 15},
        'SUP-303': {'location': 'Local', 'base_lead_time': 7}
    }
    products = ['PROD-A5', 'PROD-B8', 'PROD-C2']
    
    data = []
    start_date = datetime.date(2023, 1, 1)
    
    for _ in range(num_records):
        supplier_id = np.random.choice(list(suppliers.keys()))
        product_id = np.random.choice(products)
        order_date = start_date + datetime.timedelta(days=np.random.randint(0, 700))
        
        order_quantity = np.random.randint(100, 10000)
        production_capacity_pct = np.random.uniform(0.65, 1.0)
        
        # Check for holiday season (e.g., Jan-Feb for China, Nov-Dec for others)
        is_holiday_season = 0
        if suppliers[supplier_id]['location'] == 'China' and order_date.month in [1, 2]:
            is_holiday_season = 1
        elif order_date.month in [11, 12]:
            is_holiday_season = 1
            
        # --- Simulate Lead Time ---
        # Start with the supplier's base lead time
        lead_time = suppliers[supplier_id]['base_lead_time']
        
        # Add effects from features
        lead_time += order_quantity * 0.001  # Larger orders take slightly longer
        lead_time += production_capacity_pct * 15 # Higher capacity utilization increases lead time
        lead_time += is_holiday_season * 20 # Holiday season has a big impact
        
        # Add some random noise
        lead_time += np.random.normal(0, 3)
        lead_time = max(3, round(lead_time)) # Ensure lead time is positive
        
        data.append({
            'order_date': order_date,
            'supplier_id': supplier_id,
            'product_id': product_id,
            'supplier_location': suppliers[supplier_id]['location'],
            'order_quantity': order_quantity,
            'production_capacity_pct': round(production_capacity_pct, 2),
            'is_holiday_season': is_holiday_season,
            'lead_time_days': int(lead_time)
        })
        
    return pd.DataFrame(data)

def preprocess_data(df):
    """
    Prepares the data for machine learning by handling dates and categorical features.
    """
    # Convert 'order_date' to datetime objects
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Feature Engineering: Extract features from the date
    df['order_month'] = df['order_date'].dt.month
    df['order_day_of_week'] = df['order_date'].dt.dayofweek
    df['order_day_of_year'] = df['order_date'].dt.dayofyear
    
    # Drop the original date column as it's no longer needed
    df = df.drop('order_date', axis=1)
    
    # Convert categorical columns to the 'category' dtype for FLAML
    for col in ['supplier_id', 'product_id', 'supplier_location']:
        df[col] = df[col].astype('category')
        
    return df

# --- 1. Generate and Prepare Data ---
print("🚚 Generating synthetic supplier data...")
df = generate_supplier_data(num_records=500)
print("Sample of generated data:")
print(df.head())
print("\nData Info:")
print(df.info())


df_processed = preprocess_data(df.copy())
print("Sample of processed data:")
print(df_processed.head())
print("\nData Info:")
df_processed.info()

# Define features (X) and the target (y)
X = df_processed.drop('lead_time_days', axis=1)
y = df_processed['lead_time_days']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Configure and Run FLAML AutoML ---
print("\n🤖 Initializing FLAML AutoML for regression...")
automl = AutoML()

# Settings for the AutoML run
settings = {
    "time_budget": 60,       # Total time in seconds for the search
    "metric": 'r2',          # Primary metric to optimize for (R-squared)
    "task": 'regression',    # Specify the task type
    "log_file_name": "flaml_supplier_regression.log", # Log file
    "seed": 42,              # for reproducibility
}

# Train the AutoML model
print(f"🚀 Starting model training... (Time budget: {settings['time_budget']} seconds)")
automl.fit(X_train=X_train, y_train=y_train, **settings)

# --- 3. Evaluate the Best Model ---
print("\n✅ Training complete!")
print("--------------------------------------------------")
print(f"🏆 Best ML model found: {automl.best_estimator}")
print(f"⚙️ Best configuration: {automl.best_config}")
print(f"💯 Best R-squared score on validation data: {automl.best_loss:.4f}")
print("--------------------------------------------------")

# Make predictions on the unseen test set
y_pred = automl.predict(X_test)

# Calculate performance metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n📊 Performance on the unseen test set:")
print(f"   R-squared: {r2:.4f}")
print(f"   Mean Absolute Error (MAE): {mae:.2f} days")
print("--------------------------------------------------")


# --- 4. Predict on New User Input ---
def predict_new_order():
    """
    Takes user input for a new order and predicts the lead time.
    """
    print("\n📝 Enter details for a new order to predict lead time:")
    
    # Collect user input with validation
    supplier_id = input(f"   Enter Supplier ID {X['supplier_id'].unique().tolist()}: ")
    while supplier_id not in X['supplier_id'].unique():
        print("   Invalid supplier. Please choose from the list.")
        supplier_id = input(f"   Enter Supplier ID {X['supplier_id'].unique().tolist()}: ")

    product_id = input(f"   Enter Product ID {X['product_id'].unique().tolist()}: ")
    while product_id not in X['product_id'].unique():
        print("   Invalid product. Please choose from the list.")
        product_id = input(f"   Enter Product ID {X['product_id'].unique().tolist()}: ")

    supplier_location = input(f"   Enter Supplier Location {X['supplier_location'].unique().tolist()}: ")
    while supplier_location not in X['supplier_location'].unique():
        print("   Invalid location. Please choose from the list.")
        supplier_location = input(f"   Enter Supplier Location {X['supplier_location'].unique().tolist()}: ")
    
    order_quantity = int(input("   Enter Order Quantity (e.g., 5000): "))
    production_capacity_pct = float(input("   Enter Supplier Production Capacity (e.g., 0.95 for 95%): "))
    is_holiday_season = int(input("   Is it a holiday season? (1 for Yes, 0 for No): "))
    order_date_str = input("   Enter Order Date (YYYY-MM-DD): ")
    
    # Create a DataFrame from the input
    input_data = {
        'order_date': [pd.to_datetime(order_date_str)],
        'supplier_id': [supplier_id],
        'product_id': [product_id],
        'supplier_location': [supplier_location],
        'order_quantity': [order_quantity],
        'production_capacity_pct': [production_capacity_pct],
        'is_holiday_season': [is_holiday_season]
    }
    input_df = pd.DataFrame(input_data)
    
    # Apply the same preprocessing
    processed_input_df = preprocess_data(input_df)
    
    # Ensure columns are in the same order as training data
    processed_input_df = processed_input_df[X_train.columns]
    
    # Make the prediction
    predicted_lead_time = automl.predict(processed_input_df)
    
    print("\n--------------------------------------------------")
    print(f"🔮 Predicted Lead Time: {int(round(predicted_lead_time[0]))} days")
    print("--------------------------------------------------")

# Run the prediction function in a loop
while True:
    predict_new_order()
    again = input("Predict another order? (yes/no): ").lower()
    if again != 'yes':
        break

print("\nExiting program.")

