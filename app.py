import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib # For saving/loading models
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import logging
import json 

# Firebase Imports
import firebase_admin 
from firebase_admin import credentials, initialize_app 
from firebase_admin import firestore
from firebase_admin import auth

# Suppress Prophet logs to keep Streamlit output clean
logging.getLogger('prophet').setLevel(logging.WARNING)
# Suppress Firebase Admin SDK default logs
logging.getLogger('firebase_admin').setLevel(logging.WARNING)

# --- Configuration Constants (Directories no longer strictly needed for data persistence, but models still use them) ---
MODELS_DIR = "models"

# Paths for saved machine learning models
SALES_RF_MODEL_PATH = os.path.join(MODELS_DIR, "sales_rf_model.pkl")
CUSTOMERS_RF_MODEL_PATH = os.path.join(MODELS_DIR, "customers_rf_model.pkl")
SALES_PROPHET_MODEL_PATH = os.path.join(MODELS_DIR, "sales_prophet_model.pkl")
CUSTOMERS_PROPHET_MODEL_PATH = os.path.join(MODELS_DIR, "customers_prophet_model.pkl")

# Ensure necessary directories exist for models
os.makedirs(MODELS_DIR, exist_ok=True)


# --- Firebase Initialization and Authentication ---
@st.cache_resource(ttl=3600) # Cache the Firebase app initialization
def initialize_firebase_client():
    """Initializes Firebase Admin SDK and authenticates user."""
    
    # Get the Firebase service account from Streamlit secrets.
    # When using dotted notation in secrets.toml, this will return a streamlit.runtime.secrets.AttrDict.
    firebase_secret_value = st.secrets.get('firebase_service_account')
    
    # Use getattr for app_id and initial_auth_token as they are still runtime injections
    app_id = getattr(st, '__app_id', 'default-app-id')
    initial_auth_token = getattr(st, '__initial_auth_token', None)

    firebase_config = None # Initialize to None

    # Handle various possible types for firebase_secret_value
    if firebase_secret_value is None:
        st.error("Firebase service account configuration not found in Streamlit secrets.")
        st.info("Please go to your Streamlit Cloud app settings -> Secrets, and add your Firebase service account JSON using dotted notation (e.g., `firebase_service_account.type = '...'`).")
        return None, None, None
    elif isinstance(firebase_secret_value, str):
        # This branch is for backward compatibility if the user somehow reverted to a single string JSON.
        try:
            firebase_config = json.loads(firebase_secret_value)
        except json.JSONDecodeError as e:
            st.error(f"Error parsing Firebase service account string from secrets as JSON: {e}")
            st.info("If your secret is formatted as a single JSON string, please ensure it's valid JSON. If using dotted notation, this error should not occur with the updated code.")
            return None, None, None
    else:
        # This path is EXPECTED when using dotted notation, as Streamlit returns an AttrDict.
        # We explicitly convert it to a standard Python dict for firebase_admin.credentials.Certificate().
        try:
            firebase_config = dict(firebase_secret_value)
        except Exception as e:
            st.error(f"Error converting Streamlit secret object to dictionary: {e}")
            st.info("This usually means the secret is not a dictionary-like structure. Please ensure 'firebase_service_account' is correctly structured in your Streamlit secrets using dotted notation.")
            return None, None, None

    # Final validation that we indeed have a dictionary
    if not isinstance(firebase_config, dict) or not firebase_config:
        st.error("Firebase service account configuration is not a valid dictionary after processing secrets.")
        st.info("This indicates a fundamental issue with the secret content or Streamlit's secrets management. Please double-check your 'firebase_service_account' secret in Streamlit Cloud.")
        return None, None, None

    # --- START OF NEW PRIVATE KEY HANDLING ---
    if 'private_key' in firebase_config and isinstance(firebase_config['private_key'], str):
        # The triple-quoted string in TOML includes literal newlines.
        # However, sometimes SDKs are finicky. Let's ensure the newlines are standard '\n'.
        # This line explicitly replaces any potential escaped newlines with actual newline characters.
        # For triple-quoted strings, this might not be strictly necessary, but it acts as a safeguard.
        firebase_config['private_key'] = firebase_config['private_key'].replace('\\n', '\n')
    # --- END OF NEW PRIVATE KEY HANDLING ---

    try:
        # Check if Firebase app is already initialized
        if not firebase_admin._apps:
            # firebase_config is now guaranteed to be a standard Python dict with corrected private_key
            cred = credentials.Certificate(firebase_config) 
            initialize_app(cred) 
        
        db = firestore.client()
        
        # Authenticate
        current_user_id = None
        if initial_auth_token:
            try:
                decoded_token = auth.verify_id_token(initial_auth_token)
                current_user_id = decoded_token['uid']
                st.sidebar.success(f"Authenticated with Firebase. User ID: `{current_user_id}`") 
            except Exception as auth_e:
                st.sidebar.error(f"Authentication token verification failed: {auth_e}")
                st.sidebar.info("Falling back to anonymous user ID. If this persists, verify your token or Firebase project settings.")
                current_user_id = "anonymous_user_id"
        else:
            st.sidebar.warning("No initial auth token found. Using anonymous user ID. Data might not be private per user in collaborative contexts.")
            current_user_id = "anonymous_user_id" 

        return db, current_user_id, app_id
    except Exception as e:
        st.error(f"Error initializing Firebase or authenticating: {e}")
        st.info("Please ensure your Firebase service account key is valid and correctly formatted in Streamlit secrets. Also, verify your Firebase project settings (e.g., enabled APIs).")
        return None, None, None

db, USER_ID, APP_ID = initialize_firebase_client()

# --- Firestore Data Management Functions ---
def get_firestore_sales_collection_ref(db, user_id, app_id):
    return db.collection('artifacts').document(app_id).collection('users').document(user_id).collection('sales_data')

def get_firestore_events_collection_ref(db, user_id, app_id):
    return db.collection('artifacts').document(app_id).collection('users').document(user_id).collection('events_data')

@st.cache_data(show_spinner=False)
def load_sales_data_from_firestore(db, user_id, app_id):
    """Loads sales data from Firestore into a pandas DataFrame."""
    if not db or not user_id or not app_id:
        st.warning("Firestore is not initialized. Cannot load sales data.")
        return pd.DataFrame()

    try:
        sales_collection = get_firestore_sales_collection_ref(db, user_id, app_id)
        docs = sales_collection.stream()
        
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            if 'Date' in doc_data and 'Sales' in doc_data and 'Customers' in doc_data:
                # Firestore Timestamp to datetime
                doc_data['Date'] = doc_data['Date'].to_datetime()
                data.append(doc_data)
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce').fillna(0)
            df['Customers'] = pd.to_numeric(df['Customers'], errors='coerce').fillna(0).astype(int)
            df['Add_on_Sales'] = pd.to_numeric(df['Add_on_Sales'], errors='coerce').fillna(0)
            df = df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading sales data from Firestore: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_events_data_from_firestore(db, user_id, app_id):
    """Loads event data from Firestore into a pandas DataFrame."""
    if not db or not user_id or not app_id:
        st.warning("Firestore is not initialized. Cannot load event data.")
        return pd.DataFrame()

    try:
        events_collection = get_firestore_events_collection_ref(db, user_id, app_id)
        docs = events_collection.stream()
        
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            if 'Event_Date' in doc_data and 'Event_Name' in doc_data:
                # Firestore Timestamp to datetime
                doc_data['Event_Date'] = doc_data['Event_Date'].to_datetime()
                data.append(doc_data)
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['Event_Date'] = pd.to_datetime(df['Event_Date'])
            df = df.sort_values('Event_Date').drop_duplicates(subset=['Event_Date'], keep='last').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading events data from Firestore: {e}")
        return pd.DataFrame()

def save_sales_record_to_firestore(db, user_id, app_id, record_df):
    """Saves or updates a single sales record in Firestore."""
    if db is None or user_id is None or app_id is None:
        st.error("Firestore is not initialized. Cannot save data.")
        return False
    
    try:
        sales_collection = get_firestore_sales_collection_ref(db, user_id, app_id)
        # Use the date as the document ID for easy lookup and update
        doc_id = record_df['Date'].strftime('%Y-%m-%d')
        
        # Convert pandas Timestamp to Firestore Timestamp for saving
        record_data = record_df.to_dict(orient='records')[0] # Get dict from single-row DataFrame
        record_data['Date'] = firestore.SERVER_TIMESTAMP # This will set the server timestamp on creation/update
        # You might also store the original date as a string if exact match is needed for filtering later
        record_data['Original_Date_Str'] = doc_id 

        sales_collection.document(doc_id).set(record_data, merge=True) # merge=True updates existing fields and adds new ones
        st.cache_data.clear() # Clear cache to force reload
        return True
    except Exception as e:
        st.error(f"Error saving sales record to Firestore: {e}")
        return False

def delete_sales_record_from_firestore(db, user_id, app_id, date_str):
    """Deletes a single sales record from Firestore by date string."""
    if db is None or user_id is None or app_id is None:
        st.error("Firestore is not initialized. Cannot delete data.")
        return False
    
    try:
        sales_collection = get_firestore_sales_collection_ref(db, user_id, app_id)
        sales_collection.document(date_str).delete()
        st.cache_data.clear() # Clear cache to force reload
        return True
    except Exception as e:
        st.error(f"Error deleting sales record from Firestore: {e}")
        return False

def save_event_record_to_firestore(db, user_id, app_id, record_df):
    """Saves or updates a single event record in Firestore."""
    if db is None or user_id is None or app_id is None:
        st.error("Firestore is not initialized. Cannot save data.")
        return False
    
    try:
        events_collection = get_firestore_events_collection_ref(db, user_id, app_id)
        # Use event_date as doc_id for events too
        doc_id = record_df['Event_Date'].strftime('%Y-%m-%d')
        
        record_data = record_df.to_dict(orient='records')[0] # Get dict from single-row DataFrame
        record_data['Event_Date'] = firestore.SERVER_TIMESTAMP # Use server timestamp or convert explicitly if exact time matters
        record_data['Original_Event_Date_Str'] = doc_id # Store original date as string

        events_collection.document(doc_id).set(record_data, merge=True)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Error saving event record to Firestore: {e}")
        return False

def delete_event_record_from_firestore(db, user_id, app_id, date_str):
    """Deletes a single event record from Firestore by date string."""
    if db is None or user_id is None or app_id is None:
        st.error("Firestore is not initialized. Cannot delete data.")
        return False
    
    try:
        events_collection = get_firestore_events_collection_ref(db, user_id, app_id)
        events_collection.document(date_str).delete()
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Error deleting event record from Firestore: {e}")
        return False


# --- Preprocessing for RandomForestRegressor ---
def preprocess_rf_data(df_sales, df_events):
    """
    Preprocesses sales and events data to create features for RandomForestRegressor.
    Includes time-based features, weather one-hot encoding, and event impact.
    Handles empty input DataFrames.
    Returns X (features), y_sales (sales target), y_customers (customers target), and the processed df.
    """
    if df_sales.empty:
        return pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.DataFrame()

    df = df_sales.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Time-based features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    # Robust way to get week of year from isocalendar
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int) 
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Weather one-hot encoding
    all_weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
    for cond in all_weather_conditions:
        col_name = f'weather_{cond}'
        df[col_name] = (df['Weather'] == cond).astype(int) if 'Weather' in df.columns else 0 # Defensive check

    # Merge with events data to incorporate event impact
    df['is_event'] = 0
    df['event_impact_score'] = 0.0

    if not df_events.empty:
        df_events_copy = df_events.copy()
        df_events_copy['Event_Date'] = pd.to_datetime(df_events_copy['Event_Date'])
        impact_map = {'Low': 0.1, 'Medium': 0.5, 'High': 1.0}
        df_events_copy['Impact_Score'] = df_events_copy['Impact'].map(impact_map).fillna(0)

        # Merge based on date only
        merged = pd.merge(df[['Date']].drop_duplicates(), df_events_copy[['Event_Date', 'Impact_Score']].drop_duplicates(),
                          left_on='Date', right_on='Event_Date', how='left')
        
        # Re-merge back to original df to ensure alignment
        df = pd.merge(df, merged[['Date', 'Impact_Score']].rename(columns={'Impact_Score': 'event_impact_score_merged'}),
                      on='Date', how='left')
        df['is_event'] = (~df['event_impact_score_merged'].isna()).astype(int)
        df['event_impact_score'] = df['event_impact_score_merged'].fillna(0)
        df = df.drop(columns=['event_impact_score_merged'])


    # Lag features - fillna(0) for initial lags if not enough historical data
    df['Sales_Lag1'] = df['Sales'].shift(1).fillna(0)
    df['Customers_Lag1'] = df['Customers'].shift(1).fillna(0)
    df['Sales_Lag7'] = df['Sales'].shift(7).fillna(0)
    df['Customers_Lag7'] = df['Customers'].shift(7).fillna(0)

    # Ensure all columns are numeric before selecting features
    for col in ['Sales', 'Customers', 'Add_on_Sales']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df = df.fillna(0) # Fill any remaining NaNs, e.g., from shifting at the beginning of series

    feature_columns = [
        'day_of_week', 'day_of_year', 'month', 'year', 'week_of_year', 'is_weekend',
        'Sales_Lag1', 'Customers_Lag1', 'Sales_Lag7', 'Customers_Lag7',
        'is_event', 'event_impact_score'
    ]
    feature_columns.extend([f'weather_{cond}' for cond in all_weather_conditions])

    # Ensure all feature columns exist, adding with 0 if missing (important for consistent input to model)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_columns]
    y_sales = df['Sales']
    y_customers = df['Customers']

    # Store these in session state for use in forecasting (to maintain feature consistency)
    st.session_state['rf_feature_columns'] = feature_columns
    st.session_state['all_weather_conditions'] = all_weather_conditions

    return X, y_sales, y_customers, df

# --- Preprocessing for Prophet ---
def preprocess_prophet_data(df_sales, df_events, target_column):
    """
    Preprocesses data for the Prophet model.
    Transforms data to 'ds' (datetime) and 'y' (target) format.
    Integrates 'Add_on_Sales' and weather conditions as extra regressors, and events as holidays.
    Handles empty input DataFrames.
    """
    if df_sales.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df_sales.copy()
    df['ds'] = pd.to_datetime(df['Date'])
    df['y'] = df[target_column]

    # Ensure Add_on_Sales is always present and numeric
    if 'Add_on_Sales' in df.columns:
        df['Add_on_Sales'] = pd.to_numeric(df['Add_on_Sales'], errors='coerce').fillna(0)
    else:
        df['Add_on_Sales'] = 0.0 # Default to 0 if column is missing

    # Weather one-hot encoding for Prophet regressors
    all_weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
    for cond in all_weather_conditions:
        col_name = f'weather_{cond}'
        # Create dummy columns based on 'Weather' column; if 'Weather' is absent, these will be all zeros.
        # This ensures consistent regressor columns regardless of the input data.
        df[col_name] = (df['Weather'] == cond).astype(int) if 'Weather' in df.columns else 0

    # Prepare holidays DataFrame for Prophet from events data
    holidays_df = pd.DataFrame()
    if not df_events.empty:
        holidays_df = df_events.rename(columns={'Event_Date': 'ds', 'Event_Name': 'holiday'})
        holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
        holidays_df = holidays_df[['ds', 'holiday']].drop_duplicates(subset=['ds'])

    prophet_df = df[['ds', 'y']].copy()
    prophet_df['Add_on_Sales'] = df['Add_on_Sales'] # Copy add-on sales to the prophet df
    
    for cond in all_weather_conditions:
        col_name = f'weather_{cond}'
        prophet_df[col_name] = df[col_name] # Copy weather dummies to the prophet df

    return prophet_df, holidays_df

# --- AI Model Training Functions ---
def train_random_forest_models(X, y_sales, y_customers, n_estimators):
    """
    Trains RandomForestRegressor models for Sales and Customers, then saves them to disk.
    Requires at least 2 data points for meaningful training (due to lag features).
    """
    if X.empty or len(X) < 2:
        st.warning("Not enough data to train the RandomForest models. Need at least 2 sales records for meaningful features and training.")
        return None, None

    sales_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    customers_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)

    sales_model.fit(X, y_sales)
    joblib.dump(sales_model, SALES_RF_MODEL_PATH)

    customers_model.fit(X, y_customers)
    joblib.dump(customers_model, CUSTOMERS_RF_MODEL_PATH)

    return sales_model, customers_model

def train_prophet_models(prophet_sales_df, prophet_customers_df, holidays_df):
    """
    Trains Prophet models for Sales and Customers, then saves them to disk.
    Requires at least 2 data points for Prophet training.
    """
    if prophet_sales_df.empty or prophet_customers_df.empty:
        st.warning("Not enough data to train the Prophet models. Please add more sales records.")
        return None, None

    if len(prophet_sales_df) < 2 or len(prophet_customers_df) < 2:
        st.warning("Prophet requires at least 2 data points for training. Please add more sales records.")
        return None, None

    # Sanity check: if 'y' is all zeros for sales or customers, Prophet will struggle to learn
    # Check if target sum is zero and if there's any data
    if prophet_sales_df['y'].sum() == 0 and len(prophet_sales_df['y']) > 0:
        st.warning("Sales data for Prophet training consists only of zeros. Prophet cannot train effectively on this. Please input non-zero sales values.")
        return None, None
    if prophet_customers_df['y'].sum() == 0 and len(prophet_customers_df['y']) > 0:
        st.warning("Customer data for Prophet training consists only of zeros. Prophet cannot train effectively on this. Please input non-zero customer values.")
        return None, None
    
    # Initialize Prophet models with daily seasonality
    # Weekly and yearly seasonality are usually auto-detected by Prophet if enough data is present
    sales_prophet_model = Prophet(holidays=holidays_df, interval_width=0.95, daily_seasonality=True)
    customers_prophet_model = Prophet(holidays=holidays_df, interval_width=0.95, daily_seasonality=True)

    # Add regressors to BOTH models
    # Ensure Add_on_Sales is added if it was prepared in preprocess_prophet_data
    if 'Add_on_Sales' in prophet_sales_df.columns:
        sales_prophet_model.add_regressor('Add_on_Sales')
        customers_prophet_model.add_regressor('Add_on_Sales')
    
    # Add all possible weather conditions as regressors
    all_weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
    for cond in all_weather_conditions:
        regressor_name = f'weather_{cond}'
        # Prophet will only use regressors present in the dataframe used for .fit()
        # Adding them here ensures they are "registered" with the model
        sales_prophet_model.add_regressor(regressor_name)
        customers_prophet_model.add_regressor(regressor_name)

    # Fit the models
    sales_prophet_model.fit(prophet_sales_df)
    joblib.dump(sales_prophet_model, SALES_PROPHET_MODEL_PATH)

    customers_prophet_model.fit(prophet_customers_df)
    joblib.dump(customers_prophet_model, CUSTOMERS_PROPHET_MODEL_PATH)

    return sales_prophet_model, customers_prophet_model

@st.cache_resource(hash_funcs={pd.DataFrame: pd.util.hash_pandas_object, pd.Series: pd.util.hash_pandas_object})
def load_or_train_models_cached(model_type, n_estimators_rf=100):
    """
    Loads pre-trained models from disk if they exist, otherwise trains them.
    Models are cached to avoid retraining on every Streamlit rerun if data hasn't changed.
    """
    # Ensure Firestore is initialized before proceeding
    if db is None or USER_ID is None or APP_ID is None:
        st.warning("Firebase not fully initialized. Skipping model loading/training.")
        return None, None

    sales_df_current = load_sales_data_from_firestore(db, USER_ID, APP_ID) # Load from Firestore
    events_df_current = load_events_data_from_firestore(db, USER_ID, APP_ID) # Load from Firestore

    sales_model = None
    customers_model = None

    if model_type == "RandomForest":
        sales_model_path = SALES_RF_MODEL_PATH
        customers_model_path = CUSTOMERS_RF_MODEL_PATH
        
        if not sales_df_current.empty and sales_df_current.shape[0] >= 2:
            X, y_sales, y_customers, _ = preprocess_rf_data(sales_df_current, events_df_current)
            if not X.empty and X.shape[0] >= 2:
                if os.path.exists(sales_model_path) and os.path.exists(customers_model_path):
                    try:
                        sales_model = joblib.load(sales_model_path)
                        customers_model = joblib.load(customers_model_path)
                        st.info("RandomForest models loaded from disk.")
                    except Exception as e:
                        st.error(f"Error loading RandomForest models: {e}. Attempting to retrain.")
                        sales_model, customers_model = train_random_forest_models(X, y_sales, y_customers, n_estimators_rf)
                else:
                    st.info("No RandomForest models found on disk. Training AI models now...")
                    sales_model, customers_model = train_random_forest_models(X, y_sales, y_customers, n_estimators_rf)
            else:
                st.info("Not enough valid preprocessed data for RandomForest training (requires at least 2 records for features/lags).")
        else:
            st.info("Insufficient sales data (minimum 2 records) to train RandomForest models.")


    elif model_type == "Prophet":
        sales_model_path = SALES_PROPHET_MODEL_PATH
        customers_model_path = CUSTOMERS_PROPHET_MODEL_PATH

        if not sales_df_current.empty and sales_df_current.shape[0] >= 2:
            prophet_sales_df, holidays_df = preprocess_prophet_data(sales_df_current, events_df_current, 'Sales')
            prophet_customers_df, _ = preprocess_prophet_data(sales_df_current, events_df_current, 'Customers')

            # Ensure sales_model and customers_model are defined before checking for file existence
            sales_model_exists = os.path.exists(sales_model_path)
            customers_model_exists = os.path.exists(customers_model_path)

            if not prophet_sales_df.empty and not prophet_customers_df.empty and len(prophet_sales_df) >= 2:
                if prophet_sales_df['y'].sum() == 0 and len(prophet_sales_df['y']) > 0:
                    st.warning("Sales data for Prophet training consists only of zeros. Prophet cannot train effectively on this. Please input non-zero sales values.")
                    return None, None # Prevent training and return None models
                if prophet_customers_df['y'].sum() == 0 and len(prophet_customers_df['y']) > 0:
                    st.warning("Customer data for Prophet training consists only of zeros. Prophet cannot train effectively on this. Please input non-zero customer values.")
                    return None, None # Prevent training and return None models

                if sales_model_exists and customers_model_exists:
                    try:
                        sales_model = joblib.load(sales_model_path)
                        customers_model = joblib.load(customers_model_path)
                        st.info("Prophet models loaded from disk.")
                    except Exception as e:
                        st.error(f"Error loading Prophet models: {e}. Attempting to retrain.")
                        sales_model, customers_model = train_prophet_models(prophet_sales_df, prophet_customers_df, holidays_df)
                else:
                    st.info("No Prophet models found on disk. Training AI models now...")
                    sales_model, customers_model = train_prophet_models(prophet_sales_df, prophet_customers_df, holidays_df)
            else:
                st.info("Not enough valid preprocessed data for Prophet training (requires at least 2 records).")
        else:
            st.info("Insufficient sales data (minimum 2 records) to train Prophet models.")

    return sales_model, customers_model

# --- Forecast Generation Functions ---
def generate_rf_forecast(sales_df, events_df, sales_model, customers_model, future_weather_inputs, num_days=10):
    """
    Generates sales and customer forecasts for the next N days using RandomForest.
    Uses iterative prediction for lagged features, updating them with forecasts.
    """
    if sales_model is None or customers_model is None:
        st.warning("RandomForest models are not trained. Please ensure you have sufficient data and a model is selected and trained.")
        return pd.DataFrame()

    today = pd.Timestamp(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
    forecast_dates = [today + pd.Timedelta(days=i) for i in range(1, num_days + 1)]

    combined_data_buffer = []
    if not sales_df.empty:
        for index, row in sales_df.iterrows():
            combined_data_buffer.append({
                'Date': pd.Timestamp(row['Date']),
                'Sales': row['Sales'],
                'Customers': row['Customers']
            })
    
    combined_data_buffer.sort(key=lambda x: x['Date'])

    forecast_results = []
    
    current_sales_lag1 = combined_data_buffer[-1]['Sales'] if combined_data_buffer else 0.0
    current_customers_lag1 = combined_data_buffer[-1]['Customers'] if combined_data_buffer else 0.0

    avg_sales_history = sales_df['Sales'].mean() if not sales_df.empty else 0.0
    avg_customers_history = sales_df['Customers'].mean() if not sales_df.empty else 0.0

    for i in range(num_days):
        current_forecast_date = forecast_dates[i]
        
        current_weather_input_str = str(next((item['weather'] for item in future_weather_inputs if item['date'] == current_forecast_date.strftime('%Y-%m-%d')), 'Sunny'))

        lag7_date = current_forecast_date - pd.Timedelta(days=7)
        
        lag7_sales_val = avg_sales_history 
        lag7_customers_val = avg_customers_history 

        for entry in combined_data_buffer:
            if entry['Date'] == lag7_date:
                lag7_sales_val = entry['Sales']
                lag7_customers_val = entry['Customers']
                break
        
        current_features_data = {
            'day_of_week': current_forecast_date.weekday(),
            'day_of_year': current_forecast_date.dayofyear,
            'month': current_forecast_date.month,
            'year': current_forecast_date.year,
            'week_of_year': current_forecast_date.isocalendar()[1], 
            'is_weekend': int(current_forecast_date.weekday() in [5, 6]),
            'Sales_Lag1': current_sales_lag1,
            'Customers_Lag1': current_customers_lag1,
            'Sales_Lag7': lag7_sales_val, 
            'Customers_Lag7': lag7_customers_val, 
            'is_event': 0,
            'event_impact_score': 0.0
        }

        all_weather_conditions = st.session_state.get('all_weather_conditions', ['Sunny', 'Cloudy', 'Rainy', 'Snowy'])
        for cond in all_weather_conditions:
            current_features_data[f'weather_{cond}'] = int(current_weather_input_str == cond)

        if not events_df.empty:
            matching_event = events_df[events_df['Event_Date'].dt.date == current_forecast_date.date()]
            if not matching_event.empty:
                current_features_data['is_event'] = 1
                impact_map = {'Low': 0.1, 'Medium': 0.5, 'High': 1.0}
                current_features_data['event_impact_score'] = impact_map.get(matching_event['Impact'].iloc[0], 0)

        input_for_prediction = pd.DataFrame([current_features_data])
        
        feature_cols = st.session_state.get('rf_feature_columns', [])
        input_for_prediction = input_for_prediction.reindex(columns=feature_cols, fill_value=0)


        predicted_sales = sales_model.predict(input_for_prediction)[0]
        predicted_customers = customers_model.predict(input_for_prediction)[0]

        sales_predictions_per_tree = np.array([tree.predict(input_for_prediction)[0] for tree in sales_model.estimators_])
        customers_predictions_per_tree = np.array([tree.predict(input_for_prediction)[0] for tree in customers_model.estimators_])
        
        sales_lower = np.percentile(sales_predictions_per_tree, 2.5)
        sales_upper = np.percentile(sales_predictions_per_tree, 97.5)
        customers_lower = np.percentile(customers_predictions_per_tree, 2.5)
        customers_upper = np.percentile(customers_predictions_per_tree, 97.5)

        forecast_results.append({
            'Date': current_forecast_date.strftime('%Y-%m-%d'),
            'Forecasted Sales': max(0, round(predicted_sales, 2)),
            'Sales Lower Bound (95%)': max(0, round(sales_lower, 2)),
            'Sales Upper Bound (95%)': max(0, round(sales_upper, 2)),
            'Forecasted Customers': max(0, round(predicted_customers)),
            'Customers Lower Bound (95%)': max(0, round(customers_lower)),
            'Customers Upper Bound (95%)': max(0, round(customers_upper)),
            'Weather': current_weather_input_str 
        })

        combined_data_buffer.append({'Date': current_forecast_date, 'Sales': predicted_sales, 'Customers': predicted_customers})

        current_sales_lag1 = predicted_sales
        current_customers_lag1 = predicted_customers
        
    return pd.DataFrame(forecast_results)

def generate_prophet_forecast(sales_df, events_df, sales_model, customers_model, future_weather_inputs, num_days=10):
    """
    Generates sales and customer forecasts for the next N days using Prophet.
    """
    if sales_model is None or customers_model is None:
        st.warning("Prophet models are not trained. Please ensure you have sufficient data and a model is selected and trained.")
        return pd.DataFrame()

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    forecast_dates = [today + timedelta(days=i) for i in range(1, num_days + 1)]

    future_prophet_df = pd.DataFrame({'ds': forecast_dates})
    
    avg_add_on_sales = sales_df['Add_on_Sales'].mean() if not sales_df.empty else 0.0
    future_prophet_df['Add_on_Sales'] = avg_add_on_sales

    all_weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
    for cond in all_weather_conditions:
        col_name = f'weather_{cond}'
        future_prophet_df[col_name] = 0 # Initialize all weather columns to 0

    for i, row in future_prophet_df.iterrows():
        current_date_str = row['ds'].strftime('%Y-%m-%d')
        matching_weather_input = next((item for item in future_weather_inputs if item['date'] == current_date_str), None)
        if matching_weather_input:
            chosen_weather = matching_weather_input['weather']
            col_name = f'weather_{chosen_weather}'
            if col_name in future_prophet_df.columns:
                future_prophet_df.loc[i, col_name] = 1 # Set the chosen weather for the day to 1

    forecast_sales = sales_model.predict(future_prophet_df)
    forecast_customers = customers_model.predict(future_prophet_df)

    forecast_df = pd.DataFrame({
        'Date': forecast_sales['ds'].dt.strftime('%Y-%m-%d'),
        'Forecasted Sales': forecast_sales['yhat'].apply(lambda x: max(0, round(x, 2))),
        'Sales Lower Bound (95%)': forecast_sales['yhat_lower'].apply(lambda x: max(0, round(x, 2))),
        'Sales Upper Bound (95%)': forecast_sales['yhat_upper'].apply(lambda x: max(0, round(x, 2))),
        'Forecasted Customers': forecast_customers['yhat'].apply(lambda x: max(0, round(x))),
        'Customers Lower Bound (95%)': forecast_customers['yhat_lower'].apply(lambda x: max(0, round(x))),
        'Customers Upper Bound (95%)': forecast_customers['yhat_upper'].apply(lambda x: max(0, round(x))),
    })
    
    forecast_df['Weather'] = [next((item['weather'] for item in future_weather_inputs if item['date'] == date_str), 'Sunny') for date_str in forecast_df['Date']]

    return forecast_df

# --- Streamlit UI Layout and Logic ---
st.set_page_config(layout="wide", page_title="AI Sales & Customer Forecast App")

st.title("🎯 AI Sales & Customer Forecast Analyst")
st.markdown("Your 200 IQ analyst for daily sales and customer volume forecasting!")

# --- Initialize Streamlit Session State & Load Data ---
# Ensure Firestore is initialized before trying to load data into session state
if db is None or USER_ID is None or APP_ID is None:
    st.info("Initializing Firebase... please wait.")
    st.stop() # Stop execution until Firebase is ready

if 'app_initialized' not in st.session_state:
    st.session_state['sales_data'] = load_sales_data_from_firestore(db, USER_ID, APP_ID)
    st.session_state['events_data'] = load_events_data_from_firestore(db, USER_ID, APP_ID)
    st.session_state['sales_model'] = None
    st.session_state['customers_model'] = None
    st.session_state['model_type'] = "RandomForest"
    st.session_state['rf_n_estimators'] = 100
    st.session_state['future_weather_inputs'] = []
    st.session_state['app_initialized'] = True
    # Initial data load might trigger a rerun if there was no data before

# Display User ID
if USER_ID:
    st.sidebar.write(f"**Current User ID:** `{USER_ID}`")
else:
    st.sidebar.warning("User ID not available. Data saving might be impacted.")

# --- Initial Sample Data Creation for Events (still useful for new users) ---
def create_sample_events_if_empty_and_initialize():
    """
    Creates sample event data in Firestore if the events collection is empty.
    """
    events_df_check = load_events_data_from_firestore(db, USER_ID, APP_ID)

    if events_df_check.empty:
        st.info("No events found in Firestore. Adding sample event data...")
        sample_events = [
            {'Event_Date': pd.to_datetime('2024-06-20'), 'Event_Name': 'Annual Fair', 'Impact': 'High'},
            {'Event_Date': pd.to_datetime('2023-12-25'), 'Event_Name': 'Christmas Day', 'Impact': 'High'},
            {'Event_Date': pd.to_datetime('2024-03-15'), 'Event_Name': 'Spring Festival', 'Impact': 'Medium'},
            {'Event_Date': pd.to_datetime('2025-06-27'), 'Event_Name': 'Charter Day 2025 (Future)', 'Impact': 'High'},
            {'Event_Date': pd.to_datetime('2025-07-04'), 'Event_Name': 'Independence Day (Future)', 'Impact': 'Medium'},
            {'Event_Date': pd.to_datetime('2024-07-04'), 'Event_Name': 'Independence Day 2024', 'Impact': 'Medium'},
        ]
        success_count = 0
        for event in sample_events:
            # Firestore expects native datetime objects or Timestamps, not pandas Timestamps directly for dict conversion.
            event_for_firestore = event.copy()
            event_for_firestore['Event_Date'] = event_for_firestore['Event_Date'].to_pydatetime()
            if save_event_record_to_firestore(db, USER_ID, APP_ID, pd.DataFrame([event_for_firestore])):
                success_count += 1
        
        if success_count == len(sample_events):
            st.success("Sample event data added to Firestore! Rerunning application to load data.")
            st.cache_data.clear() # Clear cache for event data loader
            st.session_state.events_data = load_events_data_from_firestore(db, USER_ID, APP_ID) # Reload
            st.experimental_rerun()
        else:
            st.error("Failed to add all sample event data to Firestore.")
            
if not st.session_state.get('ran_sample_event_init', False) and st.session_state.events_data.empty: # Check if events_data is actually empty
    # Only run if db is initialized and there's no data
    if db and USER_ID and APP_ID:
        create_sample_events_if_empty_and_initialize()
        st.session_state['ran_sample_event_init'] = True # Set flag only after attempting creation

# --- Sidebar for Model Settings and Event Logger ---
st.sidebar.header("🛠️ Model Settings")
model_type_selection = st.sidebar.selectbox(
    "Select AI Model:",
    ["RandomForest", "Prophet"],
    index=0 if st.session_state.model_type == "RandomForest" else 1,
    key='model_type_selector',
    help="RandomForest is versatile. Prophet is good for time series with strong seasonality and holidays."
)
if model_type_selection != st.session_state.model_type:
    st.session_state.model_type = model_type_selection
    st.session_state.sales_model = None
    st.session_state.customers_model = None
    st.experimental_rerun()

if st.session_state.model_type == "RandomForest":
    rf_n_estimators = st.sidebar.slider(
        "RandomForest n_estimators:",
        min_value=50, max_value=500, value=st.session_state.rf_n_estimators, step=50,
        key='rf_n_estimators_slider',
        help="Number of trees in the forest. Higher values increase accuracy but also computation time."
    )
    if rf_n_estimators != st.session_state.rf_n_estimators:
        st.session_state.rf_n_estimators = rf_n_estimators
        st.session_state.sales_model = None
        st.session_state.customers_model = None
        st.experimental_rerun()

st.sidebar.header("🗓️ Event Logger (Past & Future)")
with st.sidebar.form("event_input_form", clear_on_submit=True):
    st.subheader("Add Historical/Future Event")
    event_date = st.date_input("Event Date", datetime.now() - timedelta(days=365), key='sidebar_event_date_input')
    event_name = st.text_input("Event Name (e.g., Charter Day, Fiesta)", max_chars=100, key='sidebar_event_name_input')
    event_impact = st.selectbox("Impact", ['Low', 'Medium', 'High'], key='sidebar_event_impact_select')
    add_event_button = st.form_submit_button("Add Event")

    if add_event_button:
        if db and USER_ID and APP_ID:
            new_event_record = {
                'Event_Date': pd.to_datetime(event_date).to_pydatetime(), # Convert to Python datetime
                'Event_Name': event_name,
                'Impact': event_impact
            }
            if save_event_record_to_firestore(db, USER_ID, APP_ID, pd.DataFrame([new_event_record])):
                st.sidebar.success(f"Event '{event_name}' added! AI will retrain.")
                st.session_state.events_data = load_events_data_from_firestore(db, USER_ID, APP_ID) # Reload
                st.session_state.sales_model = None
                st.session_state.customers_model = None
                st.experimental_rerun()
            else:
                st.sidebar.error("Failed to add event to Firestore.")
        else:
            st.sidebar.warning("Firebase not initialized. Cannot add event.")

st.sidebar.subheader("Logged Events")
# Ensure events_data is a DataFrame before sorting
if not st.session_state.events_data.empty:
    display_events_df = st.session_state.events_data.sort_values('Event_Date', ascending=False).copy()
    display_events_df['Event_Date'] = display_events_df['Event_Date'].dt.strftime('%Y-%m-%d')
    st.sidebar.dataframe(display_events_df, use_container_width=True)

    event_dates_to_delete = st.sidebar.multiselect(
        "Select events to delete:",
        st.session_state.events_data['Event_Date'].dt.strftime('%Y-%m-%d').tolist(),
        key='event_delete_multiselect'
    )
    if st.sidebar.button("Delete Selected Events", key='delete_event_btn'):
        if event_dates_to_delete:
            if db and USER_ID and APP_ID:
                all_deleted_successfully = True
                for date_str in event_dates_to_delete:
                    if not delete_event_record_from_firestore(db, USER_ID, APP_ID, date_str):
                        all_deleted_successfully = False
                        break
                if all_deleted_successfully:
                    st.sidebar.success("Selected events deleted! AI will retrain.")
                    st.session_state.events_data = load_events_data_from_firestore(db, USER_ID, APP_ID) # Reload
                    st.session_state.sales_model = None
                    st.session_state.customers_model = None
                    st.experimental_rerun()
                else:
                    st.sidebar.error("Failed to delete all selected events.")
            else:
                st.sidebar.warning("Firebase not initialized. Cannot delete events.")
        else:
            st.sidebar.warning("No events selected for deletion.")
else:
    st.sidebar.info("No events logged yet.")

# --- Model Loading and Training (runs on most reruns, but uses caching) ---
if st.session_state.sales_data.shape[0] > 1:
    with st.spinner(f"Loading/Training AI models ({st.session_state.model_type})... This happens after data changes."):
        try:
            st.session_state.sales_model, st.session_state.customers_model = load_or_train_models_cached(
                st.session_state.model_type, st.session_state.rf_n_estimators
            )
        except Exception as e:
            st.error(f"An unexpected error occurred during model loading/training: {e}")
            st.warning("Please ensure you have enough sales data (at least 2 days) and correct dependencies.")
else:
    st.info("Add more sales records (at least 2 days) to enable AI model training and forecasting. Model training will start automatically once data is sufficient.")

# --- Main Application Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Daily Sales Input", "📈 10-Day Forecast", "📊 Forecast Accuracy Tracking"])

with tab1:
    st.header("Smart Data Input System")
    st.markdown("Enter daily sales and customer data. The AI will learn from these inputs.")

    with st.form("daily_input_form", clear_on_submit=True):
        st.subheader("Add New Daily Record")
        col1, col2, col3 = st.columns(3)
        with col1:
            input_date = st.date_input("Date", datetime.now(), key='daily_input_date_picker')
        with col2:
            sales = st.number_input("Sales", min_value=0.0, format="%.2f", key='daily_sales_input')
        with col3:
            customers = st.number_input("Number of Customers", min_value=0, step=1, key='daily_customers_input')

        col4, col5 = st.columns(2)
        with col4:
            add_on_sales = st.number_input("Add-on Sales (e.g., birthdays, bulk)", min_value=0.0, format="%.2f", key='daily_addon_sales_input')
        with col5:
            weather = st.selectbox("Weather", ['Sunny', 'Cloudy', 'Rainy', 'Snowy'], key='daily_weather_select')

        add_record_button = st.form_submit_button("Add Record")

        if add_record_button:
            if db and USER_ID and APP_ID:
                new_input_record = {
                    'Date': pd.to_datetime(input_date).to_pydatetime(), # Convert to Python datetime
                    'Sales': sales,
                    'Customers': customers,
                    'Add_on_Sales': add_on_sales,
                    'Weather': weather
                }
                if save_sales_record_to_firestore(db, USER_ID, APP_ID, pd.DataFrame([new_input_record])):
                    st.success(f"Record for {input_date.strftime('%Y-%m-%d')} updated/added successfully! AI will retrain.")
                    st.session_state.sales_data = load_sales_data_from_firestore(db, USER_ID, APP_ID) # Reload
                    st.session_state.sales_model = None # Force model retraining
                    st.session_state.customers_model = None # Force model retraining
                    st.experimental_rerun() # Trigger a full rerun to update all components
                else:
                    st.error("Failed to add/update sales record to Firestore.")
            else:
                st.warning("Firebase not initialized. Cannot add record.")

    st.subheader("Most Recently Inputted/Updated Data (Last 7 Unique Days)")
    if not st.session_state.sales_data.empty:
        display_data = st.session_state.sales_data.sort_values('Date', ascending=False).drop_duplicates(subset=['Date'], keep='first').head(7).copy()
        
        if not display_data.empty:
            display_data['Date'] = display_data['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_data, use_container_width=True)
        else:
            st.info("No data manually entered or updated yet. Input new records above!")
    else:
        st.info("No data manually entered or updated yet. Input new records above!")
        

    st.subheader("Browse All Records by Month")
    if not st.session_state.sales_data.empty:
        df_all_sales = st.session_state.sales_data.copy()
        
        df_all_sales = df_all_sales.sort_values('Date').drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)

        df_all_sales['YearMonth'] = df_all_sales['Date'].dt.to_period('M')

        unique_year_months = sorted(df_all_sales['YearMonth'].unique(), reverse=True)
        
        formatted_year_months = [ym.strftime('%B %Y') for ym in unique_year_months]
        
        ym_map = {ym.strftime('%B %Y'): ym for ym in unique_year_months}

        if formatted_year_months:
            selected_ym_str = st.selectbox(
                "Select Month and Year to View:",
                options=formatted_year_months,
                key='month_year_selector'
            )
            
            selected_ym_period = ym_map[selected_ym_str]

            filtered_monthly_data = df_all_sales[
                df_all_sales['YearMonth'] == selected_ym_period
            ].sort_values('Date', ascending=False).drop('YearMonth', axis=1)

            filtered_monthly_data['Date'] = filtered_monthly_data['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(filtered_monthly_data, use_container_width=True)

            st.markdown("---")
            st.subheader(f"Edit/Delete Records for {selected_ym_str}")
            unique_dates_for_monthly_selectbox = sorted(filtered_monthly_data['Date'].tolist(), reverse=True)
            
            if unique_dates_for_monthly_selectbox:
                selected_date_str_monthly = st.selectbox(
                    "Select a record by Date for editing or deleting within this month:",
                    options=unique_dates_for_monthly_selectbox,
                    key='edit_delete_selector_monthly'
                )

                # Fetch the record for editing using the full sales data
                selected_row_df_monthly = st.session_state.sales_data[
                    st.session_state.sales_data['Date'] == pd.to_datetime(selected_date_str_monthly)
                ]
                selected_row_monthly = selected_row_df_monthly.iloc[0]

                st.markdown(f"**Selected Record for {selected_date_str_monthly}:**")
                
                edit_sales_monthly = st.number_input("Edit Sales", value=float(selected_row_monthly['Sales']), format="%.2f", key='edit_sales_input_monthly')
                edit_customers_monthly = st.number_input("Edit Customers", value=int(selected_row_monthly['Customers']), step=1, key='edit_customers_input_monthly')
                edit_add_on_sales_monthly = st.number_input("Edit Add-on Sales", value=float(selected_row_monthly['Add_on_Sales']), format="%.2f", key='edit_add_on_sales_input_monthly')
                
                weather_options_monthly = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
                try:
                    default_weather_index_monthly = weather_options_monthly.index(selected_row_monthly['Weather'])
                except ValueError:
                    default_weather_index_monthly = 0
                edit_weather_monthly = st.selectbox("Edit Weather", weather_options_monthly, index=default_weather_index_monthly, key='edit_weather_select_monthly')

                col_edit_del_btns1_monthly, col_edit_del_btns2_monthly = st.columns(2)
                
                update_button_monthly = col_edit_del_btns1_monthly.button("Update Record (Monthly View)", key='update_btn_monthly')
                delete_button_monthly = col_edit_del_btns2_monthly.button("Delete Record (Monthly View)", key='delete_btn_monthly')

                if update_button_monthly:
                    if db and USER_ID and APP_ID:
                        updated_record_data = {
                            'Date': pd.to_datetime(selected_date_str_monthly).to_pydatetime(), # Convert to Python datetime
                            'Sales': edit_sales_monthly,
                            'Customers': edit_customers_monthly,
                            'Add_on_Sales': edit_add_on_sales_monthly,
                            'Weather': edit_weather_monthly
                        }
                        if save_sales_record_to_firestore(db, USER_ID, APP_ID, pd.DataFrame([updated_record_data])):
                            st.success("Record updated successfully! AI will retrain.")
                            st.session_state.sales_data = load_sales_data_from_firestore(db, USER_ID, APP_ID) # Reload
                            st.session_state.sales_model = None
                            st.session_state.customers_model = None
                            st.experimental_rerun()
                        else:
                            st.error("Failed to update record in Firestore.")
                    else:
                        st.warning("Firebase not initialized. Cannot update record.")
                elif delete_button_monthly:
                    if db and USER_ID and APP_ID:
                        if delete_sales_record_from_firestore(db, USER_ID, APP_ID, selected_date_str_monthly):
                            st.success("Record deleted successfully! AI will retrain.")
                            st.session_state.sales_data = load_sales_data_from_firestore(db, USER_ID, APP_ID) # Reload
                            st.session_state.sales_model = None
                            st.session_state.customers_model = None
                            st.experimental_rerun()
                        else:
                            st.error("Failed to delete record from Firestore.")
                    else:
                        st.sidebar.warning("Firebase not initialized. Cannot delete record.")
            else:
                st.info("No sales records available in this month for editing or deletion.")

        else:
            st.info("No historical data available to browse by month.")
    else:
        st.info("No historical data available to browse by month.")


with tab2:
    st.header("📈 10-Day Sales & Customer Forecast")
    st.markdown("View the AI's predictions for the next 10 days.")

    st.subheader("Future Weather Forecast (What-If Scenario)")
    st.markdown("Specify the expected weather for each forecast day to see its impact. Default is 'Sunny'.")
    
    forecast_dates_for_weather = [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(10)]
    weather_options = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']

    if not st.session_state.future_weather_inputs or \
       any(item['date'] not in forecast_dates_for_weather for item in st.session_state.future_weather_inputs) or \
       len(st.session_state.future_weather_inputs) != 10:
        st.session_state.future_weather_inputs = [{'date': d, 'weather': 'Sunny'} for d in forecast_dates_for_weather]

    weather_inputs_edited = []
    for i, item in enumerate(st.session_state.future_weather_inputs):
        col_w1, col_w2 = st.columns([1, 2])
        with col_w1:
            st.write(f"**Day {i+1}: {item['date']}**")
        with col_w2:
            selected_weather = st.selectbox(
                "Weather",
                weather_options,
                index=weather_options.index(item['weather']),
                key=f"future_weather_{item['date']}"
            )
            weather_inputs_edited.append({'date': item['date'], 'weather': selected_weather})
    
    st.session_state.future_weather_inputs = weather_inputs_edited

    if st.button("Generate 10-Day Forecast", key='generate_forecast_btn'):
        if st.session_state.sales_data.empty or st.session_state.sales_data.shape[0] < 2:
            st.warning("Please enter at least 2 days of sales data to generate a meaningful forecast.")
        elif st.session_state.sales_model is None or st.session_state.customers_model is None:
             st.warning("AI models are not ready. Please ensure you have sufficient data and the models are trained.")
        else:
            with st.spinner(f"Generating forecast using {st.session_state.model_type}... This might take a moment as the AI thinks ahead!"):
                if st.session_state.model_type == "RandomForest":
                    forecast_df = generate_rf_forecast(
                        st.session_state.sales_data,
                        st.session_state.events_data,
                        st.session_state.sales_model,
                        st.session_state.customers_model,
                        st.session_state.future_weather_inputs
                    )
                elif st.session_state.model_type == "Prophet":
                    forecast_df = generate_prophet_forecast(
                        st.session_state.sales_data,
                        st.session_state.events_data,
                        st.session_state.sales_model,
                        st.session_state.customers_model,
                        st.session_state.future_weather_inputs
                    )
                st.session_state.forecast_df = forecast_df
                st.success("Forecast generated!")
    
    if 'forecast_df' in st.session_state and not st.session_state.forecast_df.empty:
        st.subheader("Forecasted Data (with 95% Confidence Interval)")
        st.dataframe(st.session_state.forecast_df, use_container_width=True)

        csv = st.session_state.forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast as CSV",
            data=csv,
            file_name="sales_customer_forecast.csv",
            mime="text/csv",
        )

        st.subheader("Forecast Visualization")
        historical_df_plot = st.session_state.sales_data.copy()
        historical_df_plot['Date'] = pd.to_datetime(historical_df_plot['Date'])

        forecast_df_plot = st.session_state.forecast_df.copy()
        forecast_df_plot['Date'] = pd.to_datetime(forecast_df_plot['Date'])

        fig_sales, ax_sales = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=historical_df_plot, x='Date', y='Sales', label='Actual Sales', marker='o', ax=ax_sales, color='blue')
        sns.lineplot(data=forecast_df_plot, x='Date', y='Forecasted Sales', label='Forecasted Sales', marker='x', linestyle='--', ax=ax_sales, color='red')
        
        ax_sales.fill_between(forecast_df_plot['Date'], forecast_df_plot['Sales Lower Bound (95%)'], forecast_df_plot['Sales Upper Bound (95%)'], color='red', alpha=0.2, label='95% Confidence Interval')

        ax_sales.set_title(f'Sales: Actual vs. Forecast ({st.session_state.model_type} Model)')
        ax_sales.set_xlabel('Date')
        ax_sales.set_ylabel('Sales')
        ax_sales.legend()
        ax_sales.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_sales)

        fig_customers, ax_customers = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=historical_df_plot, x='Date', y='Customers', label='Actual Customers', marker='o', ax=ax_customers, color='blue')
        sns.lineplot(data=forecast_df_plot, x='Date', y='Forecasted Customers', label='Forecasted Customers', marker='x', linestyle='--', ax=ax_customers, color='red')
        
        ax_customers.fill_between(forecast_df_plot['Date'], forecast_df_plot['Customers Lower Bound (95%)'], forecast_df_plot['Customers Upper Bound (95%)'], color='red', alpha=0.2, label='95% Confidence Interval')

        ax_customers.set_title(f'Customers: Actual vs. Forecast ({st.session_state.model_type} Model)')
        ax_customers.set_xlabel('Date')
        ax_customers.set_ylabel('Customers')
        ax_customers.legend()
        ax_customers.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_customers)
    else:
        st.info("Click 'Generate 10-Day Forecast' to see predictions.")


with tab3:
    st.header("Forecast Accuracy Tracking")
    st.markdown("Compare past forecasts with actual data to track AI performance.")

    if st.button("Calculate Accuracy", key='calculate_accuracy_btn'):
        if st.session_state.sales_data.empty:
            st.warning("No sales data available to calculate accuracy.")
        elif st.session_state.sales_data.shape[0] < 2:
            st.warning("Please enter at least 2 days of sales data to calculate accuracy.")
        elif st.session_state.sales_model is None or st.session_state.customers_model is None:
             st.warning("AI models are not ready. Please ensure you have sufficient data and the models are trained first.")
        else:
            with st.spinner(f"Calculating accuracy using {st.session_state.model_type}... This may take a moment."):
                if st.session_state.model_type == "RandomForest":
                    X_hist, y_sales_hist, y_customers_hist, _ = preprocess_rf_data(st.session_state.sales_data, st.session_state.events_data)
                    
                    if not X_hist.empty and X_hist.shape[0] > 1:
                        predicted_sales_hist = st.session_state.sales_model.predict(X_hist)
                        predicted_customers_hist = st.session_state.customers_model.predict(X_hist)

                        accuracy_plot_df = pd.DataFrame({
                            'Date': st.session_state.sales_data['Date'].iloc[X_hist.index],
                            'Actual Sales': y_sales_hist,
                            'Predicted Sales': predicted_sales_hist,
                            'Actual Customers': y_customers_hist,
                            'Predicted Customers': predicted_customers_hist
                        })

                        mae_sales = mean_absolute_error(accuracy_plot_df['Actual Sales'], accuracy_plot_df['Predicted Sales'])
                        r2_sales = r2_score(accuracy_plot_df['Actual Sales'], accuracy_plot_df['Predicted Sales'])

                        mae_customers = mean_absolute_error(accuracy_plot_df['Actual Customers'], accuracy_plot_df['Predicted Customers'])
                        r2_customers = r2_score(accuracy_plot_df['Actual Customers'], accuracy_plot_df['Predicted Customers'])
                        
                        st.subheader(f"Overall Model Accuracy ({st.session_state.model_type})")
                        st.write(f"**Sales MAE (Mean Absolute Error):** {mae_sales:.2f}")
                        st.write(f"**Sales R² Score:** {r2_sales:.2f}")
                        st.write(f"**Customers MAE (Mean Absolute Error):** {mae_customers:.2f}")
                        st.write(f"**Customers R² Score:** {r2_customers:.2f}")
                        st.info("An R² score closer to 1 indicates a better fit. MAE shows average error in units.")

                        # --- Feature Importance Visualization for RandomForest ---
                        st.subheader("RandomForest Feature Importance")
                        if st.session_state.sales_model is not None and st.session_state.customers_model is not None and 'rf_feature_columns' in st.session_state and st.session_state['rf_feature_columns']:
                            feature_importances = pd.DataFrame({
                                'Feature': st.session_state['rf_feature_columns'],
                                'Sales Importance': st.session_state.sales_model.feature_importances_,
                                'Customers Importance': st.session_state.customers_model.feature_importances_
                            })
                            feature_importances = feature_importances.sort_values(by='Sales Importance', ascending=False)

                            fig_fi_sales, ax_fi_sales = plt.subplots(figsize=(10, 6))
                            sns.barplot(x='Sales Importance', y='Feature', data=feature_importances.head(10), ax=ax_fi_sales, palette='viridis')
                            ax_fi_sales.set_title("Top 10 Most Important Features for Sales Forecast")
                            ax_fi_sales.set_xlabel("Importance (Relative)")
                            ax_fi_sales.set_ylabel("Feature")
                            plt.tight_layout()
                            st.pyplot(fig_fi_sales)

                            fig_fi_customers, ax_fi_customers = plt.subplots(figsize=(10, 6))
                            sns.barplot(x='Customers Importance', y='Feature', data=feature_importances.sort_values(by='Customers Importance', ascending=False).head(10), ax=ax_fi_customers, palette='plasma')
                            ax_fi_customers.set_title("Top 10 Most Important Features for Customer Forecast")
                            ax_fi_customers.set_xlabel("Importance (Relative)")
                            ax_fi_customers.set_ylabel("Feature")
                            plt.tight_layout()
                            st.pyplot(fig_fi_customers)
                        else:
                            st.info("Feature importance will be displayed here after the RandomForest models are trained with sufficient data.")
                        # --- End Feature Importance ---

                        fig_acc_sales, ax_acc_sales = plt.subplots(figsize=(12, 6))
                        sns.lineplot(data=accuracy_plot_df, x='Date', y='Actual Sales', label='Actual Sales', marker='o', ax=ax_acc_sales)
                        sns.lineplot(data=accuracy_plot_df, x='Date', y='Predicted Sales', label='Predicted Sales', marker='x', linestyle='--', ax=ax_acc_sales)
                        ax_acc_sales.set_title(f'Historical Sales: Actual vs. Predicted ({st.session_state.model_type})')
                        ax_acc_sales.set_xlabel('Date')
                        ax_acc_sales.set_ylabel('Sales')
                        ax_acc_sales.legend()
                        ax_acc_sales.grid(True)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig_acc_sales)

                        fig_acc_customers, ax_acc_customers = plt.subplots(figsize=(12, 6))
                        sns.lineplot(data=accuracy_plot_df, x='Date', y='Actual Customers', label='Actual Customers', marker='o', ax=ax_acc_customers)
                        sns.lineplot(data=accuracy_plot_df, x='Date', y='Predicted Customers', label='Predicted Customers', marker='x', linestyle='--', ax=ax_acc_customers)
                        ax_acc_customers.set_title(f'Historical Customers: Actual vs. Predicted ({st.session_state.model_type})')
                        ax_acc_customers.set_xlabel('Date')
                        ax_acc_customers.set_ylabel('Customers')
                        ax_acc_customers.legend()
                        ax_acc_customers.grid(True)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig_acc_customers)
                    else:
                        st.warning("Not enough data points after preprocessing for accuracy calculation. Please add more sales records.")

                elif st.session_state.model_type == "Prophet":
                    if st.session_state.sales_data.shape[0] < 30:
                        st.warning("Prophet cross-validation requires at least 30 days of historical data for meaningful results. Please add more records.")
                    else:
                        st.info("Running Prophet cross-validation. This might take a while for large datasets.")
                        
                        sales_prophet_df_cv, holidays_df_cv = preprocess_prophet_data(st.session_state.sales_data, st.session_state.events_data, 'Sales')
                        customers_prophet_df_cv, _ = preprocess_prophet_data(st.session_state.sales_data, st.session_state.events_data, 'Customers')

                        if sales_prophet_df_cv.empty or customers_prophet_df_cv.empty or len(sales_prophet_df_cv) < 2:
                            st.warning("Prophet preprocessed data is empty or insufficient. Cannot run cross-validation.")
                        elif sales_prophet_df_cv['y'].sum() == 0 and len(sales_prophet_df_cv['y']) > 0:
                            st.warning("Prophet cross-validation cannot run with sales data consisting only of zeros. Please input non-zero values.")
                        elif customers_prophet_df_cv['y'].sum() == 0 and len(customers_prophet_df_cv['y']) > 0:
                             st.warning("Prophet cross-validation cannot run with customer data consisting only of zeros. Please input non-zero values.")
                        else:
                            try:
                                with st.spinner("Performing cross-validation for Sales model..."):
                                    df_cv_sales = cross_validation(
                                        st.session_state.sales_model, initial='30 days', period='15 days', horizon='10 days'
                                    )
                                with st.spinner("Calculating performance metrics for Sales..."):
                                    df_p_sales = performance_metrics(df_cv_sales)
                                
                                with st.spinner("Performing cross-validation for Customers model..."):
                                    df_cv_customers = cross_validation(
                                        st.session_state.customers_model, initial='30 days', period='15 days', horizon='10 days'
                                    )
                                with st.spinner("Calculating performance metrics for Customers..."):
                                    df_p_customers = performance_metrics(df_cv_customers)

                                st.subheader(f"Prophet Model Performance Metrics (Cross-Validation)")
                                st.write("Sales Model Performance:")
                                st.dataframe(df_p_sales.head(), use_container_width=True)
                                st.write("Customers Model Performance:")
                                st.dataframe(df_p_customers.head(), use_container_width=True)
                                st.info("Metrics are calculated over various forecast horizons. MAE and RMSE are typically desired to be lower.")

                                fig_sales_rmse = plot_cross_validation_metric(df_cv_sales, metric='rmse')
                                fig_sales_mae = plot_cross_validation_metric(df_cv_sales, metric='mae')
                                fig_customers_rmse = plot_cross_validation_metric(df_cv_customers, metric='rmse')
                                fig_customers_mae = plot_cross_validation_metric(df_cv_customers, metric='mae')

                                fig_sales_rmse.update_layout(title_text='Sales: RMSE vs. Horizon')
                                fig_sales_mae.update_layout(title_text='Sales: MAE vs. Horizon')
                                fig_customers_rmse.update_layout(title_text='Customers: RMSE vs. Horizon')
                                fig_customers_mae.update_layout(title_text='Customers: MAE vs. Horizon')

                                st.subheader("Prophet Cross-Validation Plots")
                                st.write("Sales RMSE:")
                                st.pyplot(fig_sales_rmse)
                                st.write("Sales MAE:")
                                st.pyplot(fig_sales_mae)
                                st.write("Customers RMSE:")
                                st.pyplot(fig_customers_rmse)
                                st.write("Customers MAE:")
                                st.pyplot(fig_customers_mae)

                            except Exception as e:
                                st.error(f"Error during Prophet cross-validation: {e}. Ensure sufficient data and model setup.")
    else:
        st.info("Click 'Calculate Accuracy' to see how well the AI performs on past data.")
