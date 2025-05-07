import pandas as pd
import numpy as np
import streamlit as st

def load_data():
    """
    Load the Enterprise Catering dataset
    
    Returns:
    --------
    pandas.DataFrame
        The raw dataset
    """
    try:
        # Try to load the data from the local file
        df = pd.read_csv("Enterprise Event & Catering Data Jan 2023 through February 2025(New- Events and Functions).csv", encoding='latin1')
    except FileNotFoundError:
        # If file is not found, use Streamlit file uploader
        st.warning("Data file not found. Please upload the data file.")
        uploaded_file = st.file_uploader("Upload Enterprise Event & Catering Data CSV", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            st.error("No data file uploaded. Cannot proceed.")
            st.stop()
    
    return df

def preprocess_data(df):
    """
    Preprocess the data for analysis and modeling
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The raw dataset
        
    Returns:
    --------
    pandas.DataFrame
        The preprocessed dataset
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Remove columns with high missing values
    df.drop('Lkup_Space_Description', axis=1, inplace=True)
    
    # Convert Event Description to lowercase
    df['Event_Description'] = df['Event_Description'].astype(str).str.lower()
    
    # Convert OrderedAttendance to numeric
    df['OrderedAttendance'] = pd.to_numeric(df['OrderedAttendance'], errors='coerce').astype('Int64')
    
    # Filter out negative attendance
    df = df[df['OrderedAttendance'] >= 0]
    
    # Convert date columns
    date_cols = ['StartDate', 'EndDate', 'FunctionStart', 'FunctionEnd']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Convert numeric columns
    numerical_cols = ['ActualRevenue', 'OrderedRevenue', 'ForecastRevenue', 'OrderedAttendance']
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    # Clean string columns
    object_cols = df.select_dtypes(include='object').columns.tolist()
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()
    
    # Replace "nan" with actual NaN
    cols_to_fix = ['Space', 'Space_Description', 'Event_Contact_Name']
    for col in cols_to_fix:
        df[col] = df[col].replace('nan', pd.NA)
    
    # Remove outliers from numerical features (values above 99th percentile)
    for col in numerical_cols:
        q99 = df[col].quantile(0.99)
        if pd.notna(q99):
             df = df[df[col] <= q99]
    
    # Filter data by date range
    df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce', utc=True)
    df['EndDate'] = pd.to_datetime(df['EndDate'], errors='coerce', utc=True)
    
    start_min = pd.Timestamp('2023-01-01', tz='UTC')
    end_max = pd.Timestamp('2025-06-30', tz='UTC')
    
    mask = ((df['StartDate'] >= start_min) & (df['EndDate'] <= end_max))
    df = df[mask].reset_index(drop=True)
    
    return df

def create_features(time_series_df, target_col, date_col):
    """
    Create features for time series forecasting
    
    Parameters:
    -----------
    time_series_df : pandas.DataFrame
        DataFrame with at least a date column and a target column
    target_col : str
        Name of the target column
    date_col : str
        Name of the date column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional features for forecasting
    """
    # Create a copy to avoid modifying the original
    df = time_series_df.copy()
    
    # Ensure date column is datetime and has consistent timezone
    df[date_col] = pd.to_datetime(df[date_col])
    # Add UTC timezone if not already timezone-aware
    if df[date_col].dt.tz is None:
        df[date_col] = df[date_col].dt.tz_localize('UTC')
    
    # Create lag features (1, 3 months)
    for lag in [1, 3]:
        if len(df) > lag:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Create rolling window features
    if len(df) >= 3:
        df['MA_3'] = df[target_col].rolling(window=3).mean()
        df['Vol_3'] = df[target_col].rolling(window=3).std()
    
    # Create momentum feature
    if len(df) >= 2:
        df['Momentum_1'] = df[target_col].pct_change(periods=1)
    
    # Extract date features
    df['Year'] = df[date_col].dt.year
    df['Month_Num'] = df[date_col].dt.month
    
    # Create cyclical features for seasonality
    df['sin_month'] = np.sin(2 * np.pi * df['Month_Num'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['Month_Num'] / 12)
    
    # Drop rows with missing values (from lag/rolling features)
    df = df.dropna().reset_index(drop=True)
    
    return df

def create_prediction_features(df, forecast_horizon, target_col):
    """
    Create feature matrix for prediction of future periods
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Historical data with features
    forecast_horizon : int
        Number of periods to forecast
    target_col : str
        Name of the target column
        
    Returns:
    --------
    pandas.DataFrame
        Feature matrix for prediction
    """
    # Get the last date in the dataset
    last_date = df['Month'].max()
    
    # Get the timezone from the last date (if it exists)
    tz = last_date.tz
    
    # Create future dates
    future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(forecast_horizon)]
    
    # Ensure consistent timezone
    if tz is not None:
        future_dates = [date.tz_localize(tz) if date.tz is None else date.tz_convert(tz) for date in future_dates]
    
    # Initialize the prediction features dataframe
    pred_features = pd.DataFrame({'Month': future_dates})
    
    # Add date-based features
    pred_features['Year'] = pred_features['Month'].dt.year
    pred_features['Month_Num'] = pred_features['Month'].dt.month
    pred_features['sin_month'] = np.sin(2 * np.pi * pred_features['Month_Num'] / 12)
    pred_features['cos_month'] = np.cos(2 * np.pi * pred_features['Month_Num'] / 12)
    
    # For each future period, create the appropriate lag features
    for i in range(forecast_horizon):
        for lag in [1, 3]:
            lag_col = f'lag_{lag}'
            if lag_col in df.columns:
                if i < lag:
                    required_history_offset = lag - i
                    if len(df[target_col]) >= required_history_offset:
                        pred_features.loc[i, lag_col] = df[target_col].iloc[-required_history_offset]
                    else:
                        pred_features.loc[i, lag_col] = np.nan
                else:
                    # Placeholder for recursive fill later, or use actual if not CIs
                    pred_features.loc[i, lag_col] = np.nan
    
    # Add placeholders for other features
    if 'MA_3' in df.columns:
        pred_features['MA_3'] = np.nan
    
    if 'Vol_3' in df.columns:
        pred_features['Vol_3'] = np.nan
    
    if 'Momentum_1' in df.columns:
        pred_features['Momentum_1'] = np.nan
        
    return pred_features 
