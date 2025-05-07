import pandas as pd
import numpy as np
import random
from xgboost import XGBRegressor
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_revenue_model(revenue_df, forecast_horizon=4, n_trials=50, n_bootstraps=100):
    """
    Train an XGBoost model for revenue forecasting with Optuna hyperparameter optimization
    and generate forecasts with confidence intervals using bootstrap.
    
    Parameters:
    -----------
    revenue_df : pandas.DataFrame
        DataFrame with the following columns: 'Month', 'ActualRevenue', and feature columns
    forecast_horizon : int
        Number of months to forecast ahead
    n_trials : int
        Number of trials for Optuna hyperparameter optimization
    n_bootstraps : int
        Number of bootstrap iterations for confidence interval
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with forecasted values and confidence intervals
    """
    # Split data into features and target
    X = revenue_df.drop(columns=['Month', 'ActualRevenue'])
    y = revenue_df['ActualRevenue']
    
    if len(X) == 0:
        # Not enough data to train, return empty forecast
        last_date = pd.Timestamp.now(tz='UTC') # Use a sensible default if revenue_df is empty
        if not revenue_df.empty and 'Month' in revenue_df.columns and not revenue_df['Month'].empty:
            last_date = revenue_df['Month'].max()
        
        tz = last_date.tz
        future_months = [last_date + pd.DateOffset(months=i+1) for i in range(forecast_horizon)]
        if tz is not None:
            future_months = [date.tz_localize(tz) if date.tz is None else date.tz_convert(tz) for date in future_months]
            
        return pd.DataFrame({
            'Month': future_months,
            'ForecastedRevenue': [np.nan] * forecast_horizon,
            'Lower_CI': [np.nan] * forecast_horizon,
            'Upper_CI': [np.nan] * forecast_horizon
        })

    # Train-test split (use last 3 months as test, or fewer if not enough data)
    n_test = min(3, len(X) - 1) if len(X) > 1 else 0
    X_train, X_test = X.iloc[:-n_test] if n_test > 0 else X, X.iloc[-n_test:] if n_test > 0 else pd.DataFrame()
    y_train, y_test = y.iloc[:-n_test] if n_test > 0 else y, y.iloc[-n_test:] if n_test > 0 else pd.Series()
    
    # Define Optuna objective function
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1),
        }
        
        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=42
        )
        if not X_train.empty and not y_train.empty:
            model.fit(X_train, y_train)
            if not X_test.empty and not y_test.empty:
                preds = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                return rmse
        return float('inf') # Return high RMSE if training/testing is not possible
    
    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")
    # Reduce trials if dataset is very small to prevent Optuna errors
    actual_n_trials = n_trials if len(X_train) >= n_test + 1 and n_test > 0 else max(1, n_trials // 5)
    study.optimize(objective, n_trials=actual_n_trials, show_progress_bar=False) # Turn off progress bar for Streamlit
    best_params = study.best_params
    
    # Train final model on full dataset (available for training)
    final_model = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=42
    )
    final_model.fit(X, y)
    
    # Generate forecasts for future periods
    last_date = revenue_df['Month'].max()
    tz = last_date.tz
    future_months = [last_date + pd.DateOffset(months=i+1) for i in range(forecast_horizon)]
    if tz is not None:
        future_months = [date.tz_localize(tz) if date.tz is None else date.tz_convert(tz) for date in future_months]
    
    forecast_df = pd.DataFrame({
        'Month': future_months,
        'ForecastedRevenue': [np.nan] * forecast_horizon, # Placeholder, will be filled by recursive forecast
        'Lower_CI': [np.nan] * forecast_horizon,
        'Upper_CI': [np.nan] * forecast_horizon
    })
    
    # --- Recursive Forecasting with the main model ---
    main_model_preds = np.zeros(forecast_horizon)
    temp_y_for_pred = y.copy() 

    for step in range(forecast_horizon):
        month = future_months[step]
        current_features = { 
            'Year': month.year,
            'Month_Num': month.month,
            'sin_month': np.sin(2 * np.pi * month.month / 12),
            'cos_month': np.cos(2 * np.pi * month.month / 12)
        }
        for lag_val in [1, 2, 3, 6, 12]:
            lag_col_name = f'lag_{lag_val}'
            if lag_col_name in X.columns: 
                if step < lag_val:
                    required_history_offset = lag_val - step
                    if len(temp_y_for_pred) >= required_history_offset:
                        current_features[lag_col_name] = temp_y_for_pred.iloc[-required_history_offset]
                    else:
                        current_features[lag_col_name] = np.nan 
                else:
                    current_features[lag_col_name] = main_model_preds[step - lag_val]
        
        series_for_rolling = pd.concat([temp_y_for_pred, pd.Series(main_model_preds[:step])], ignore_index=True)
        if 'MA_3' in X.columns:
            current_features['MA_3'] = series_for_rolling.iloc[-3:].mean() if len(series_for_rolling) >=1 else np.nan
        if 'Vol_3' in X.columns:
            current_features['Vol_3'] = series_for_rolling.iloc[-3:].std() if len(series_for_rolling) >=1 else np.nan
        if 'Momentum_1' in X.columns:
            current_features['Momentum_1'] = (series_for_rolling.iloc[-1] - series_for_rolling.iloc[-2]) / series_for_rolling.iloc[-2] if len(series_for_rolling) >=2 else np.nan
        
        pred_input_df = pd.DataFrame([current_features])[X.columns] 
        prediction = final_model.predict(pred_input_df)[0]
        main_model_preds[step] = prediction

    forecast_df['ForecastedRevenue'] = main_model_preds
    # --- End of Recursive Forecasting with the main model ---

    # Dynamic block size for bootstrap
    block_size = min(12, len(X)) if len(X) > 0 else 1 # Ensure block_size is at least 1 if X is not empty
    
    if len(X) > 0 and len(X) >= block_size: # Check if X is not empty and has enough data for at least one block
        boot_preds_ci = np.zeros((n_bootstraps, forecast_horizon))
        blocks = []
        # Ensure loop for blocks can run: len(X) - block_size + 1 must be > 0
        if len(X) >= block_size:
            for i in range(len(X) - block_size + 1):
                X_block = X.iloc[i:i+block_size]
                y_block = y.iloc[i:i+block_size]
                blocks.append((X_block, y_block))
        
        if blocks: # Proceed only if blocks were actually created
            k_val = max(1, int(np.ceil(len(X)/block_size))) # Ensure enough blocks are sampled if len(X) is small
            for b in range(n_bootstraps):
                random.seed(b)
                sampled_blocks = random.choices(blocks, k=k_val)
                
                X_boot = pd.concat([block[0] for block in sampled_blocks]).reset_index(drop=True)
                y_boot = pd.concat([block[1] for block in sampled_blocks]).reset_index(drop=True)
                
                boot_model = XGBRegressor(**best_params, objective="reg:squarederror", random_state=b)
                boot_model.fit(X_boot, y_boot)
                
                current_boot_preds = np.zeros(forecast_horizon)
                temp_y_boot_for_pred = y_boot.copy() 

                for step in range(forecast_horizon):
                    month = future_months[step]
                    current_features_boot = { 
                        'Year': month.year, 'Month_Num': month.month,
                        'sin_month': np.sin(2 * np.pi * month.month / 12),
                        'cos_month': np.cos(2 * np.pi * month.month / 12)
                    }
                    for lag_val in [1, 2, 3, 6, 12]:
                        lag_col_name = f'lag_{lag_val}'
                        if lag_col_name in X.columns:
                            if step < lag_val:
                                required_history_offset = lag_val - step
                                if len(temp_y_boot_for_pred) >= required_history_offset:
                                    current_features_boot[lag_col_name] = temp_y_boot_for_pred.iloc[-required_history_offset]
                                else:
                                    current_features_boot[lag_col_name] = np.nan
                            else:
                                current_features_boot[lag_col_name] = current_boot_preds[step - lag_val]
                    
                    series_for_rolling_boot = pd.concat([temp_y_boot_for_pred, pd.Series(current_boot_preds[:step])], ignore_index=True)
                    if 'MA_3' in X.columns:
                        current_features_boot['MA_3'] = series_for_rolling_boot.iloc[-3:].mean() if len(series_for_rolling_boot) >=1 else np.nan
                    if 'Vol_3' in X.columns:
                        current_features_boot['Vol_3'] = series_for_rolling_boot.iloc[-3:].std() if len(series_for_rolling_boot) >=1 else np.nan
                    if 'Momentum_1' in X.columns:
                        current_features_boot['Momentum_1'] = (series_for_rolling_boot.iloc[-1] - series_for_rolling_boot.iloc[-2]) / series_for_rolling_boot.iloc[-2] if len(series_for_rolling_boot) >=2 else np.nan

                    pred_input_df_boot = pd.DataFrame([current_features_boot])[X.columns]
                    prediction_boot = boot_model.predict(pred_input_df_boot)[0]
                    current_boot_preds[step] = prediction_boot
                boot_preds_ci[b, :] = current_boot_preds

            forecast_df['Lower_CI'] = np.percentile(boot_preds_ci, 5, axis=0)
            forecast_df['Upper_CI'] = np.percentile(boot_preds_ci, 95, axis=0)
    
    return forecast_df

def train_event_count_model(event_df, forecast_horizon=4, n_trials=50, n_bootstraps=100):
    """
    Train an XGBoost model for event count forecasting with Optuna hyperparameter optimization
    and generate forecasts with confidence intervals using bootstrap.
    
    Parameters:
    -----------
    event_df : pandas.DataFrame
        DataFrame with the following columns: 'Month', 'Event_Count', and feature columns
    forecast_horizon : int
        Number of months to forecast ahead
    n_trials : int
        Number of trials for Optuna hyperparameter optimization
    n_bootstraps : int
        Number of bootstrap iterations for confidence interval
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with forecasted values and confidence intervals
    """
    # Split data into features and target
    X = event_df.drop(columns=['Month', 'Event_Count'])
    y = event_df['Event_Count']
    
    if len(X) == 0:
        last_date = pd.Timestamp.now(tz='UTC') 
        if not event_df.empty and 'Month' in event_df.columns and not event_df['Month'].empty:
            last_date = event_df['Month'].max()
        tz = last_date.tz
        future_months = [last_date + pd.DateOffset(months=i+1) for i in range(forecast_horizon)]
        if tz is not None:
            future_months = [date.tz_localize(tz) if date.tz is None else date.tz_convert(tz) for date in future_months]
        return pd.DataFrame({
            'Month': future_months,
            'ForecastedRevenue': [np.nan] * forecast_horizon, # Renamed for consistency in app.py
            'Lower_CI': [np.nan] * forecast_horizon,
            'Upper_CI': [np.nan] * forecast_horizon
        })

    n_test = min(3, len(X) - 1) if len(X) > 1 else 0
    X_train, X_test = X.iloc[:-n_test] if n_test > 0 else X, X.iloc[-n_test:] if n_test > 0 else pd.DataFrame()
    y_train, y_test = y.iloc[:-n_test] if n_test > 0 else y, y.iloc[-n_test:] if n_test > 0 else pd.Series()
    
    # Define Optuna objective function
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1),
        }
        
        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=42
        )
        if not X_train.empty and not y_train.empty:
            model.fit(X_train, y_train)
            if not X_test.empty and not y_test.empty:
                preds = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                return rmse
        return float('inf')
    
    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")
    actual_n_trials = n_trials if len(X_train) >= n_test + 1 and n_test > 0 else max(1, n_trials // 5)
    study.optimize(objective, n_trials=actual_n_trials, show_progress_bar=False)
    best_params = study.best_params
    
    # Train final model on full dataset
    final_model = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=42
    )
    final_model.fit(X, y)
    
    # Generate forecasts for future periods
    last_date = event_df['Month'].max()
    
    # Get the timezone from the last date (if it exists)
    tz = last_date.tz
    
    # Create future months with the same timezone as the input data
    future_months = [last_date + pd.DateOffset(months=i+1) for i in range(forecast_horizon)]
    
    # Ensure consistent timezone
    if tz is not None:
        future_months = [date.tz_localize(tz) if date.tz is None else date for date in future_months]
    
    # Create dataframe to store forecasts
    forecast_df = pd.DataFrame({
        'Month': future_months,
        'ForecastedRevenue': [np.nan] * forecast_horizon, # Renamed for consistency
        'Lower_CI': [np.nan] * forecast_horizon,
        'Upper_CI': [np.nan] * forecast_horizon
    })
    
    main_model_preds = np.zeros(forecast_horizon)
    temp_y_for_pred = y.copy()

    for step in range(forecast_horizon):
        month = future_months[step]
        current_features = {
            'Year': month.year, 'Month_Num': month.month,
            'sin_month': np.sin(2 * np.pi * month.month / 12),
            'cos_month': np.cos(2 * np.pi * month.month / 12)
        }
        for lag_val in [1, 2, 3, 6, 12]:
            lag_col_name = f'lag_{lag_val}'
            if lag_col_name in X.columns:
                if step < lag_val:
                    required_history_offset = lag_val - step
                    if len(temp_y_for_pred) >= required_history_offset:
                        current_features[lag_col_name] = temp_y_for_pred.iloc[-required_history_offset]
                    else:
                        current_features[lag_col_name] = np.nan
                else:
                    current_features[lag_col_name] = main_model_preds[step - lag_val]
        
        series_for_rolling = pd.concat([temp_y_for_pred, pd.Series(main_model_preds[:step])], ignore_index=True)
        if 'MA_3' in X.columns:
            current_features['MA_3'] = series_for_rolling.iloc[-3:].mean() if len(series_for_rolling) >=1 else np.nan
        if 'Vol_3' in X.columns:
            current_features['Vol_3'] = series_for_rolling.iloc[-3:].std() if len(series_for_rolling) >=1 else np.nan
        if 'Momentum_1' in X.columns:
            current_features['Momentum_1'] = (series_for_rolling.iloc[-1] - series_for_rolling.iloc[-2]) / series_for_rolling.iloc[-2] if len(series_for_rolling) >=2 else np.nan
        
        pred_input_df = pd.DataFrame([current_features])[X.columns]
        prediction = final_model.predict(pred_input_df)[0]
        main_model_preds[step] = prediction

    forecast_df['ForecastedRevenue'] = main_model_preds # Using 'ForecastedRevenue' for consistency

    block_size = min(12, len(X)) if len(X) > 0 else 1
    if len(X) > 0 and len(X) >= block_size:
        boot_preds_ci = np.zeros((n_bootstraps, forecast_horizon))
        blocks = []
        if len(X) >= block_size:
            for i in range(len(X) - block_size + 1):
                X_block = X.iloc[i:i+block_size]
                y_block = y.iloc[i:i+block_size]
                blocks.append((X_block, y_block))
        
        if blocks:
            k_val = max(1, int(np.ceil(len(X)/block_size)))
            for b in range(n_bootstraps):
                random.seed(b)
                sampled_blocks = random.choices(blocks, k=k_val)
                X_boot = pd.concat([block[0] for block in sampled_blocks]).reset_index(drop=True)
                y_boot = pd.concat([block[1] for block in sampled_blocks]).reset_index(drop=True)
                boot_model = XGBRegressor(**best_params, objective="reg:squarederror", random_state=b)
                boot_model.fit(X_boot, y_boot)
                
                current_boot_preds = np.zeros(forecast_horizon)
                temp_y_boot_for_pred = y_boot.copy()
                for step in range(forecast_horizon):
                    month = future_months[step]
                    current_features_boot = {
                        'Year': month.year, 'Month_Num': month.month,
                        'sin_month': np.sin(2 * np.pi * month.month / 12),
                        'cos_month': np.cos(2 * np.pi * month.month / 12)
                    }
                    for lag_val in [1, 2, 3, 6, 12]:
                        lag_col_name = f'lag_{lag_val}'
                        if lag_col_name in X.columns:
                            if step < lag_val:
                                required_history_offset = lag_val - step
                                if len(temp_y_boot_for_pred) >= required_history_offset:
                                    current_features_boot[lag_col_name] = temp_y_boot_for_pred.iloc[-required_history_offset]
                                else:
                                    current_features_boot[lag_col_name] = np.nan
                            else:
                                current_features_boot[lag_col_name] = current_boot_preds[step - lag_val]
                    
                    series_for_rolling_boot = pd.concat([temp_y_boot_for_pred, pd.Series(current_boot_preds[:step])], ignore_index=True)
                    if 'MA_3' in X.columns:
                        current_features_boot['MA_3'] = series_for_rolling_boot.iloc[-3:].mean() if len(series_for_rolling_boot) >=1 else np.nan
                    if 'Vol_3' in X.columns:
                        current_features_boot['Vol_3'] = series_for_rolling_boot.iloc[-3:].std() if len(series_for_rolling_boot) >=1 else np.nan
                    if 'Momentum_1' in X.columns:
                        current_features_boot['Momentum_1'] = (series_for_rolling_boot.iloc[-1] - series_for_rolling_boot.iloc[-2]) / series_for_rolling_boot.iloc[-2] if len(series_for_rolling_boot) >=2 else np.nan
                    
                    pred_input_df_boot = pd.DataFrame([current_features_boot])[X.columns]
                    prediction_boot = boot_model.predict(pred_input_df_boot)[0]
                    current_boot_preds[step] = prediction_boot
                boot_preds_ci[b, :] = current_boot_preds
            
            forecast_df['Lower_CI'] = np.percentile(boot_preds_ci, 5, axis=0)
            forecast_df['Upper_CI'] = np.percentile(boot_preds_ci, 95, axis=0)
    
    return forecast_df

def train_catering_model(catering_df, forecast_horizon=4, n_trials=50, n_bootstraps=100):
    """
    Train an XGBoost model for catering event count forecasting with Optuna hyperparameter optimization
    and generate forecasts with confidence intervals using bootstrap.
    
    Parameters:
    -----------
    catering_df : pandas.DataFrame
        DataFrame with the following columns: 'Month', 'Event_Count', and feature columns
    forecast_horizon : int
        Number of months to forecast ahead
    n_trials : int
        Number of trials for Optuna hyperparameter optimization
    n_bootstraps : int
        Number of bootstrap iterations for confidence interval
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with forecasted values and confidence intervals
    """
    # Split data into features and target
    X = catering_df.drop(columns=['Month', 'Event_Count'])
    y = catering_df['Event_Count']
    
    if len(X) == 0:
        last_date = pd.Timestamp.now(tz='UTC') 
        if not catering_df.empty and 'Month' in catering_df.columns and not catering_df['Month'].empty:
            last_date = catering_df['Month'].max()
        tz = last_date.tz
        future_months = [last_date + pd.DateOffset(months=i+1) for i in range(forecast_horizon)]
        if tz is not None:
            future_months = [date.tz_localize(tz) if date.tz is None else date.tz_convert(tz) for date in future_months]
        return pd.DataFrame({
            'Month': future_months,
            'ForecastedRevenue': [np.nan] * forecast_horizon, # Renamed for consistency
            'Lower_CI': [np.nan] * forecast_horizon,
            'Upper_CI': [np.nan] * forecast_horizon
        })

    n_test = min(3, len(X) - 1) if len(X) > 1 else 0
    X_train, X_test = X.iloc[:-n_test] if n_test > 0 else X, X.iloc[-n_test:] if n_test > 0 else pd.DataFrame()
    y_train, y_test = y.iloc[:-n_test] if n_test > 0 else y, y.iloc[-n_test:] if n_test > 0 else pd.Series()
    
    # Define Optuna objective function
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1),
        }
        
        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=42
        )
        if not X_train.empty and not y_train.empty:
            model.fit(X_train, y_train)
            if not X_test.empty and not y_test.empty:
                preds = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                return rmse
        return float('inf')
    
    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")
    actual_n_trials = n_trials if len(X_train) >= n_test + 1 and n_test > 0 else max(1, n_trials // 5)
    study.optimize(objective, n_trials=actual_n_trials, show_progress_bar=False)
    best_params = study.best_params
    
    # Train final model on full dataset
    final_model = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=42
    )
    final_model.fit(X, y)
    
    # Generate forecasts for future periods
    last_date = catering_df['Month'].max()
    
    # Get the timezone from the last date (if it exists)
    tz = last_date.tz
    
    # Create future months with the same timezone as the input data
    future_months = [last_date + pd.DateOffset(months=i+1) for i in range(forecast_horizon)]
    
    # Ensure consistent timezone
    if tz is not None:
        future_months = [date.tz_localize(tz) if date.tz is None else date for date in future_months]
    
    # Create dataframe to store forecasts
    forecast_df = pd.DataFrame({
        'Month': future_months,
        'ForecastedRevenue': [np.nan] * forecast_horizon, # Renamed for consistency
        'Lower_CI': [np.nan] * forecast_horizon,
        'Upper_CI': [np.nan] * forecast_horizon
    })
    
    main_model_preds = np.zeros(forecast_horizon)
    temp_y_for_pred = y.copy()

    for step in range(forecast_horizon):
        month = future_months[step]
        current_features = {
            'Year': month.year, 'Month_Num': month.month,
            'sin_month': np.sin(2 * np.pi * month.month / 12),
            'cos_month': np.cos(2 * np.pi * month.month / 12)
        }
        for lag_val in [1, 2, 3, 6, 12]:
            lag_col_name = f'lag_{lag_val}'
            if lag_col_name in X.columns:
                if step < lag_val:
                    required_history_offset = lag_val - step
                    if len(temp_y_for_pred) >= required_history_offset:
                        current_features[lag_col_name] = temp_y_for_pred.iloc[-required_history_offset]
                    else:
                        current_features[lag_col_name] = np.nan
                else:
                    current_features[lag_col_name] = main_model_preds[step - lag_val]
        
        series_for_rolling = pd.concat([temp_y_for_pred, pd.Series(main_model_preds[:step])], ignore_index=True)
        if 'MA_3' in X.columns:
            current_features['MA_3'] = series_for_rolling.iloc[-3:].mean() if len(series_for_rolling) >=1 else np.nan
        if 'Vol_3' in X.columns:
            current_features['Vol_3'] = series_for_rolling.iloc[-3:].std() if len(series_for_rolling) >=1 else np.nan
        if 'Momentum_1' in X.columns:
            current_features['Momentum_1'] = (series_for_rolling.iloc[-1] - series_for_rolling.iloc[-2]) / series_for_rolling.iloc[-2] if len(series_for_rolling) >=2 else np.nan
        
        pred_input_df = pd.DataFrame([current_features])[X.columns]
        prediction = final_model.predict(pred_input_df)[0]
        main_model_preds[step] = prediction

    forecast_df['ForecastedRevenue'] = main_model_preds # Using 'ForecastedRevenue' for consistency

    block_size = min(12, len(X)) if len(X) > 0 else 1
    if len(X) > 0 and len(X) >= block_size:
        boot_preds_ci = np.zeros((n_bootstraps, forecast_horizon))
        blocks = []
        if len(X) >= block_size:
            for i in range(len(X) - block_size + 1):
                X_block = X.iloc[i:i+block_size]
                y_block = y.iloc[i:i+block_size]
                blocks.append((X_block, y_block))
        
        if blocks:
            k_val = max(1, int(np.ceil(len(X)/block_size)))
            for b in range(n_bootstraps):
                random.seed(b)
                sampled_blocks = random.choices(blocks, k=k_val)
                X_boot = pd.concat([block[0] for block in sampled_blocks]).reset_index(drop=True)
                y_boot = pd.concat([block[1] for block in sampled_blocks]).reset_index(drop=True)
                boot_model = XGBRegressor(**best_params, objective="reg:squarederror", random_state=b)
                boot_model.fit(X_boot, y_boot)
                
                current_boot_preds = np.zeros(forecast_horizon)
                temp_y_boot_for_pred = y_boot.copy()
                for step in range(forecast_horizon):
                    month = future_months[step]
                    current_features_boot = {
                        'Year': month.year, 'Month_Num': month.month,
                        'sin_month': np.sin(2 * np.pi * month.month / 12),
                        'cos_month': np.cos(2 * np.pi * month.month / 12)
                    }
                    for lag_val in [1, 2, 3, 6, 12]:
                        lag_col_name = f'lag_{lag_val}'
                        if lag_col_name in X.columns:
                            if step < lag_val:
                                required_history_offset = lag_val - step
                                if len(temp_y_boot_for_pred) >= required_history_offset:
                                    current_features_boot[lag_col_name] = temp_y_boot_for_pred.iloc[-required_history_offset]
                                else:
                                    current_features_boot[lag_col_name] = np.nan
                            else:
                                current_features_boot[lag_col_name] = current_boot_preds[step - lag_val]
                    
                    series_for_rolling_boot = pd.concat([temp_y_boot_for_pred, pd.Series(current_boot_preds[:step])], ignore_index=True)
                    if 'MA_3' in X.columns:
                        current_features_boot['MA_3'] = series_for_rolling_boot.iloc[-3:].mean() if len(series_for_rolling_boot) >=1 else np.nan
                    if 'Vol_3' in X.columns:
                        current_features_boot['Vol_3'] = series_for_rolling_boot.iloc[-3:].std() if len(series_for_rolling_boot) >=1 else np.nan
                    if 'Momentum_1' in X.columns:
                        current_features_boot['Momentum_1'] = (series_for_rolling_boot.iloc[-1] - series_for_rolling_boot.iloc[-2]) / series_for_rolling_boot.iloc[-2] if len(series_for_rolling_boot) >=2 else np.nan
                    
                    pred_input_df_boot = pd.DataFrame([current_features_boot])[X.columns]
                    prediction_boot = boot_model.predict(pred_input_df_boot)[0]
                    current_boot_preds[step] = prediction_boot
                boot_preds_ci[b, :] = current_boot_preds
            
            forecast_df['Lower_CI'] = np.percentile(boot_preds_ci, 5, axis=0)
            forecast_df['Upper_CI'] = np.percentile(boot_preds_ci, 95, axis=0)
            
    return forecast_df 