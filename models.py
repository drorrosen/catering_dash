import pandas as pd
import numpy as np
import random
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define default parameters for Standard XGBoost mode
DEFAULT_XGB_PARAMS_FOR_STANDARD_MODE = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0.1,
    "objective": "reg:squarederror", # Ensure objective is included
    "random_state": 42,
    "n_jobs": -1
}

# Define default parameters for Lower RF mode
DEFAULT_RF_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1,
    "max_depth": 10, # Added a sensible default max_depth
    "min_samples_leaf": 5 # Added a sensible default min_samples_leaf
}

def train_revenue_model(revenue_df, forecast_horizon, forecast_mode="lower_rf", n_trials=50, n_bootstraps=100):
    """
    Train a model for revenue forecasting.
    Supports 'lower_rf' (RandomForest default, no CIs), 
             'standard_xgb' (XGBoost default, CIs), and 
             'advanced_xgb' (XGBoost Optuna, CIs).
    """
    target_column_name = 'ActualRevenue'
    model_to_use = None
    best_params = {}
    effective_n_bootstraps = 0
    run_optuna = False

    X = revenue_df.drop(columns=['Month', target_column_name])
    y = revenue_df[target_column_name]

    if len(X) == 0:
        # Handle empty data case (common to all modes)
        last_date = pd.Timestamp.now(tz='UTC') 
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

    # Determine parameters and model type based on mode
    if forecast_mode == "lower_rf":
        model_to_use = RandomForestRegressor
        best_params = DEFAULT_RF_PARAMS.copy()
        effective_n_bootstraps = n_bootstraps
        run_optuna = False
    elif forecast_mode == "standard_xgb":
        model_to_use = XGBRegressor
        best_params = DEFAULT_XGB_PARAMS_FOR_STANDARD_MODE.copy()
        effective_n_bootstraps = n_bootstraps
        run_optuna = False
    elif forecast_mode == "advanced_xgb":
        model_to_use = XGBRegressor
        # Optuna will determine best_params later
        effective_n_bootstraps = n_bootstraps
        run_optuna = True
    else:
        raise ValueError(f"Invalid forecast_mode: {forecast_mode}. Choose 'lower_rf', 'standard_xgb', or 'advanced_xgb'.")

    # Common train-test split needed for Optuna if applicable
    n_test = min(3, len(X) - 1) if len(X) > 1 else 0
    X_train, X_test = X.iloc[:-n_test] if n_test > 0 else X, X.iloc[-n_test:] if n_test > 0 else pd.DataFrame()
    y_train, y_test = y.iloc[:-n_test] if n_test > 0 else y, y.iloc[-n_test:] if n_test > 0 else pd.Series()

    # Run Optuna only if mode is advanced_xgb
    if run_optuna:
        if model_to_use != XGBRegressor: # Should not happen with current modes, but good check
             raise ValueError("Optuna optimization is only configured for XGBoost ('advanced_xgb' mode).")
             
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 2, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 1),
                "objective": "reg:squarederror", # Keep objective
                "random_state": 42, # Keep random state
                "n_jobs": -1
            }
            model = XGBRegressor(**params)
            if not X_train.empty and not y_train.empty:
                model.fit(X_train, y_train)
                if not X_test.empty and not y_test.empty:
                    preds = model.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    return rmse
            return float('inf')
        
        study = optuna.create_study(direction="minimize")
        actual_n_trials = n_trials if len(X_train) >= n_test + 1 and n_test > 0 else max(1, n_trials // 5)
        study.optimize(objective, n_trials=actual_n_trials, show_progress_bar=False)
        best_params = study.best_params
        # Ensure necessary params for XGB are present after Optuna
        best_params['objective'] = "reg:squarederror"
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1

    # Train final model on the full dataset using determined params and model type
    final_model = model_to_use(**best_params)
    final_model.fit(X, y)
    
    # --- Generate forecasts (common logic, uses final_model) ---
    last_date = revenue_df['Month'].max()
    tz = last_date.tz
    future_months = [last_date + pd.DateOffset(months=i+1) for i in range(forecast_horizon)]
    if tz is not None:
        future_months = [date.tz_localize(tz) if date.tz is None else date.tz_convert(tz) for date in future_months]
    
    forecast_df = pd.DataFrame({
        'Month': future_months,
        'ForecastedRevenue': [np.nan] * forecast_horizon,
        'Lower_CI': [np.nan] * forecast_horizon,
        'Upper_CI': [np.nan] * forecast_horizon
    })
    
    main_model_preds = np.zeros(forecast_horizon)
    temp_y_for_pred = y.copy()
    feature_cols_ordered = list(X.columns) # Get feature order once

    for step in range(forecast_horizon):
        month = future_months[step]
        current_features = { 
            'Year': month.year,
            'Month_Num': month.month,
            'sin_month': np.sin(2 * np.pi * month.month / 12),
            'cos_month': np.cos(2 * np.pi * month.month / 12)
        }
        # Generate lags based on past actuals and previous forecasts
        series_with_forecasts = pd.concat([temp_y_for_pred, pd.Series(main_model_preds[:step])], ignore_index=True)
        for lag_val in [1, 2, 3, 6, 12]:
            lag_col_name = f'lag_{lag_val}'
            if lag_col_name in feature_cols_ordered:
                 if len(series_with_forecasts) >= lag_val:
                     current_features[lag_col_name] = series_with_forecasts.iloc[-lag_val]
                 else:
                     current_features[lag_col_name] = np.nan
        
        # Generate rolling features based on past actuals and previous forecasts
        if 'MA_3' in feature_cols_ordered:
            current_features['MA_3'] = series_with_forecasts.iloc[-3:].mean() if len(series_with_forecasts) >=3 else np.nan
        if 'Vol_3' in feature_cols_ordered:
            current_features['Vol_3'] = series_with_forecasts.iloc[-3:].std() if len(series_with_forecasts) >=3 else np.nan
        if 'Momentum_1' in feature_cols_ordered:
             if len(series_with_forecasts) >= 2:
                 mom_val = (series_with_forecasts.iloc[-1] - series_with_forecasts.iloc[-2]) / series_with_forecasts.iloc[-2] if series_with_forecasts.iloc[-2] != 0 else 0
                 current_features['Momentum_1'] = mom_val
             else: 
                 current_features['Momentum_1'] = np.nan
        
        pred_input_df = pd.DataFrame([current_features])[feature_cols_ordered] # Ensure order
        prediction = final_model.predict(pred_input_df)[0]
        main_model_preds[step] = prediction

    forecast_df['ForecastedRevenue'] = main_model_preds

    # --- Bootstrapping for CI (only if effective_n_bootstraps > 0) ---
    block_size = min(12, len(X)) if len(X) > 0 else 1 
    
    if effective_n_bootstraps > 0 and len(X) > 0 and len(X) >= block_size:
        boot_preds_ci = np.zeros((effective_n_bootstraps, forecast_horizon))
        blocks = []
        if len(X) >= block_size:
            for i in range(len(X) - block_size + 1):
                X_block = X.iloc[i:i+block_size]
                y_block = y.iloc[i:i+block_size]
                blocks.append((X_block, y_block))
        
        if blocks: 
            k_val = max(1, int(np.ceil(len(X)/block_size))) 
            for b in range(effective_n_bootstraps):
                random.seed(b)
                sampled_blocks = random.choices(blocks, k=k_val)
                
                X_boot = pd.concat([block[0] for block in sampled_blocks]).reset_index(drop=True)
                y_boot = pd.concat([block[1] for block in sampled_blocks]).reset_index(drop=True)
                
                # Use the same model type and params as the main model for bootstrapping
                # Important: Create a *new* instance for each bootstrap iteration
                boot_model_params = best_params.copy()
                if 'random_state' in boot_model_params: boot_model_params['random_state'] = b # Vary random state for bootstrapping
                if model_to_use == XGBRegressor and 'n_jobs' in boot_model_params: boot_model_params['n_jobs'] = 1 # Avoid nested parallelism issues
                if model_to_use == RandomForestRegressor and 'n_jobs' in boot_model_params: boot_model_params['n_jobs'] = 1

                boot_model = model_to_use(**boot_model_params)
                boot_model.fit(X_boot, y_boot)
                
                current_boot_preds = np.zeros(forecast_horizon)
                temp_y_boot_for_pred = y_boot.copy() 

                for step_b in range(forecast_horizon): 
                    month_b = future_months[step_b] 
                    current_features_boot = { 
                        'Year': month_b.year, 'Month_Num': month_b.month,
                        'sin_month': np.sin(2 * np.pi * month_b.month / 12),
                        'cos_month': np.cos(2 * np.pi * month_b.month / 12)
                    }
                    # Generate lags based on bootstrapped history and previous forecasts
                    series_with_boot_forecasts = pd.concat([temp_y_boot_for_pred, pd.Series(current_boot_preds[:step_b])], ignore_index=True)
                    for lag_val in [1, 2, 3, 6, 12]:
                        lag_col_name = f'lag_{lag_val}'
                        if lag_col_name in feature_cols_ordered:
                             if len(series_with_boot_forecasts) >= lag_val:
                                 current_features_boot[lag_col_name] = series_with_boot_forecasts.iloc[-lag_val]
                             else:
                                 current_features_boot[lag_col_name] = np.nan
                    
                    # Generate rolling features based on bootstrapped history and previous forecasts
                    if 'MA_3' in feature_cols_ordered:
                        current_features_boot['MA_3'] = series_with_boot_forecasts.iloc[-3:].mean() if len(series_with_boot_forecasts) >=3 else np.nan
                    if 'Vol_3' in feature_cols_ordered:
                        current_features_boot['Vol_3'] = series_with_boot_forecasts.iloc[-3:].std() if len(series_with_boot_forecasts) >=3 else np.nan
                    if 'Momentum_1' in feature_cols_ordered:
                         if len(series_with_boot_forecasts) >= 2:
                             mom_val_boot = (series_with_boot_forecasts.iloc[-1] - series_with_boot_forecasts.iloc[-2]) / series_with_boot_forecasts.iloc[-2] if series_with_boot_forecasts.iloc[-2] != 0 else 0
                             current_features_boot['Momentum_1'] = mom_val_boot
                         else: 
                             current_features_boot['Momentum_1'] = np.nan

                    pred_input_df_boot = pd.DataFrame([current_features_boot])[feature_cols_ordered]
                    prediction_boot = boot_model.predict(pred_input_df_boot)[0]
                    current_boot_preds[step_b] = prediction_boot
                boot_preds_ci[b, :] = current_boot_preds

            forecast_df['Lower_CI'] = np.percentile(boot_preds_ci, 5, axis=0)
            forecast_df['Upper_CI'] = np.percentile(boot_preds_ci, 95, axis=0)
    
    return forecast_df

def train_event_count_model(event_df, forecast_horizon, forecast_mode="lower_rf", n_trials=50, n_bootstraps=100):
    """
    Train a model for event count forecasting.
    Supports 'lower_rf' (RandomForest default, no CIs), 
             'standard_xgb' (XGBoost default, CIs), and 
             'advanced_xgb' (XGBoost Optuna, CIs).
    """
    target_column_name = 'Event_Count'
    model_to_use = None
    best_params = {}
    effective_n_bootstraps = 0
    run_optuna = False

    X = event_df.drop(columns=['Month', target_column_name])
    y = event_df[target_column_name]

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
            'ForecastedRevenue': [np.nan] * forecast_horizon, 
            'Lower_CI': [np.nan] * forecast_horizon,
            'Upper_CI': [np.nan] * forecast_horizon
        })

    if forecast_mode == "lower_rf":
        model_to_use = RandomForestRegressor
        best_params = DEFAULT_RF_PARAMS.copy()
        effective_n_bootstraps = n_bootstraps
        run_optuna = False
    elif forecast_mode == "standard_xgb":
        model_to_use = XGBRegressor
        best_params = DEFAULT_XGB_PARAMS_FOR_STANDARD_MODE.copy()
        effective_n_bootstraps = n_bootstraps
        run_optuna = False
    elif forecast_mode == "advanced_xgb":
        model_to_use = XGBRegressor
        effective_n_bootstraps = n_bootstraps
        run_optuna = True
    else:
        raise ValueError(f"Invalid forecast_mode: {forecast_mode}. Choose 'lower_rf', 'standard_xgb', or 'advanced_xgb'.")

    n_test = min(3, len(X) - 1) if len(X) > 1 else 0
    X_train, X_test = X.iloc[:-n_test] if n_test > 0 else X, X.iloc[-n_test:] if n_test > 0 else pd.DataFrame()
    y_train, y_test = y.iloc[:-n_test] if n_test > 0 else y, y.iloc[-n_test:] if n_test > 0 else pd.Series()

    if run_optuna:
        if model_to_use != XGBRegressor:
             raise ValueError("Optuna optimization is only configured for XGBoost ('advanced_xgb' mode).")
        def objective(trial):
             params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 2, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 1),
                "objective": "reg:squarederror",
                "random_state": 42,
                "n_jobs": -1
             }
             model = XGBRegressor(**params)
             if not X_train.empty and not y_train.empty:
                 model.fit(X_train, y_train)
                 if not X_test.empty and not y_test.empty:
                     preds = model.predict(X_test)
                     rmse = np.sqrt(mean_squared_error(y_test, preds))
                     return rmse
             return float('inf')
        study = optuna.create_study(direction="minimize")
        actual_n_trials = n_trials if len(X_train) >= n_test + 1 and n_test > 0 else max(1, n_trials // 5)
        study.optimize(objective, n_trials=actual_n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_params['objective'] = "reg:squarederror"
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1

    final_model = model_to_use(**best_params)
    final_model.fit(X, y)
    
    last_date = event_df['Month'].max()
    tz = last_date.tz
    future_months = [last_date + pd.DateOffset(months=i+1) for i in range(forecast_horizon)]
    if tz is not None:
        future_months = [date.tz_localize(tz) if date.tz is None else date for date in future_months]
    
    forecast_df = pd.DataFrame({
        'Month': future_months,
        'ForecastedRevenue': [np.nan] * forecast_horizon, 
        'Lower_CI': [np.nan] * forecast_horizon,
        'Upper_CI': [np.nan] * forecast_horizon
    })
    
    main_model_preds = np.zeros(forecast_horizon)
    temp_y_for_pred = y.copy()
    feature_cols_ordered = list(X.columns)

    for step in range(forecast_horizon):
        month = future_months[step]
        current_features = {
            'Year': month.year, 'Month_Num': month.month,
            'sin_month': np.sin(2 * np.pi * month.month / 12),
            'cos_month': np.cos(2 * np.pi * month.month / 12)
        }
        series_with_forecasts = pd.concat([temp_y_for_pred, pd.Series(main_model_preds[:step])], ignore_index=True)
        for lag_val in [1, 2, 3, 6, 12]:
            lag_col_name = f'lag_{lag_val}'
            if lag_col_name in feature_cols_ordered:
                 if len(series_with_forecasts) >= lag_val:
                     current_features[lag_col_name] = series_with_forecasts.iloc[-lag_val]
                 else:
                     current_features[lag_col_name] = np.nan
        if 'MA_3' in feature_cols_ordered:
            current_features['MA_3'] = series_with_forecasts.iloc[-3:].mean() if len(series_with_forecasts) >=3 else np.nan
        if 'Vol_3' in feature_cols_ordered:
            current_features['Vol_3'] = series_with_forecasts.iloc[-3:].std() if len(series_with_forecasts) >=3 else np.nan
        if 'Momentum_1' in feature_cols_ordered:
             if len(series_with_forecasts) >= 2:
                 mom_val = (series_with_forecasts.iloc[-1] - series_with_forecasts.iloc[-2]) / series_with_forecasts.iloc[-2] if series_with_forecasts.iloc[-2] != 0 else 0
                 current_features['Momentum_1'] = mom_val
             else: 
                 current_features['Momentum_1'] = np.nan
        
        pred_input_df = pd.DataFrame([current_features])[feature_cols_ordered]
        prediction = final_model.predict(pred_input_df)[0]
        main_model_preds[step] = prediction

    forecast_df['ForecastedRevenue'] = main_model_preds

    block_size = min(12, len(X)) if len(X) > 0 else 1
    if effective_n_bootstraps > 0 and len(X) > 0 and len(X) >= block_size:
        boot_preds_ci = np.zeros((effective_n_bootstraps, forecast_horizon))
        blocks = []
        if len(X) >= block_size:
            for i in range(len(X) - block_size + 1):
                X_block = X.iloc[i:i+block_size]
                y_block = y.iloc[i:i+block_size]
                blocks.append((X_block, y_block))
        
        if blocks:
            k_val = max(1, int(np.ceil(len(X)/block_size)))
            for b in range(effective_n_bootstraps):
                random.seed(b)
                sampled_blocks = random.choices(blocks, k=k_val)
                X_boot = pd.concat([block[0] for block in sampled_blocks]).reset_index(drop=True)
                y_boot = pd.concat([block[1] for block in sampled_blocks]).reset_index(drop=True)
                
                boot_model_params = best_params.copy()
                if 'random_state' in boot_model_params: boot_model_params['random_state'] = b
                if model_to_use == XGBRegressor and 'n_jobs' in boot_model_params: boot_model_params['n_jobs'] = 1
                if model_to_use == RandomForestRegressor and 'n_jobs' in boot_model_params: boot_model_params['n_jobs'] = 1
                
                boot_model = model_to_use(**boot_model_params)
                boot_model.fit(X_boot, y_boot)
                
                current_boot_preds = np.zeros(forecast_horizon)
                temp_y_boot_for_pred = y_boot.copy()
                for step_b in range(forecast_horizon):
                    month_b = future_months[step_b]
                    current_features_boot = {
                        'Year': month_b.year, 'Month_Num': month_b.month,
                        'sin_month': np.sin(2 * np.pi * month_b.month / 12),
                        'cos_month': np.cos(2 * np.pi * month_b.month / 12)
                    }
                    series_with_boot_forecasts = pd.concat([temp_y_boot_for_pred, pd.Series(current_boot_preds[:step_b])], ignore_index=True)
                    for lag_val in [1, 2, 3, 6, 12]:
                        lag_col_name = f'lag_{lag_val}'
                        if lag_col_name in feature_cols_ordered:
                             if len(series_with_boot_forecasts) >= lag_val:
                                 current_features_boot[lag_col_name] = series_with_boot_forecasts.iloc[-lag_val]
                             else:
                                 current_features_boot[lag_col_name] = np.nan
                    if 'MA_3' in feature_cols_ordered:
                        current_features_boot['MA_3'] = series_with_boot_forecasts.iloc[-3:].mean() if len(series_with_boot_forecasts) >=3 else np.nan
                    if 'Vol_3' in feature_cols_ordered:
                        current_features_boot['Vol_3'] = series_with_boot_forecasts.iloc[-3:].std() if len(series_with_boot_forecasts) >=3 else np.nan
                    if 'Momentum_1' in feature_cols_ordered:
                         if len(series_with_boot_forecasts) >= 2:
                             mom_val_boot = (series_with_boot_forecasts.iloc[-1] - series_with_boot_forecasts.iloc[-2]) / series_with_boot_forecasts.iloc[-2] if series_with_boot_forecasts.iloc[-2] != 0 else 0
                             current_features_boot['Momentum_1'] = mom_val_boot
                         else: 
                             current_features_boot['Momentum_1'] = np.nan
                    
                    pred_input_df_boot = pd.DataFrame([current_features_boot])[feature_cols_ordered]
                    prediction_boot = boot_model.predict(pred_input_df_boot)[0]
                    current_boot_preds[step_b] = prediction_boot
                boot_preds_ci[b, :] = current_boot_preds
            
            forecast_df['Lower_CI'] = np.percentile(boot_preds_ci, 5, axis=0)
            forecast_df['Upper_CI'] = np.percentile(boot_preds_ci, 95, axis=0)
    
    return forecast_df

def train_catering_model(catering_df, forecast_horizon, forecast_mode="lower_rf", n_trials=50, n_bootstraps=100):
    """
    Train a model for catering event count forecasting.
    Supports 'lower_rf' (RandomForest default, no CIs), 
             'standard_xgb' (XGBoost default, CIs), and 
             'advanced_xgb' (XGBoost Optuna, CIs).
    """
    target_column_name = 'Event_Count' # For catering model
    model_to_use = None
    best_params = {}
    effective_n_bootstraps = 0
    run_optuna = False

    X = catering_df.drop(columns=['Month', target_column_name])
    y = catering_df[target_column_name]
    
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
            'ForecastedRevenue': [np.nan] * forecast_horizon, 
            'Lower_CI': [np.nan] * forecast_horizon,
            'Upper_CI': [np.nan] * forecast_horizon
        })

    if forecast_mode == "lower_rf":
        model_to_use = RandomForestRegressor
        best_params = DEFAULT_RF_PARAMS.copy()
        effective_n_bootstraps = n_bootstraps
        run_optuna = False
    elif forecast_mode == "standard_xgb":
        model_to_use = XGBRegressor
        best_params = DEFAULT_XGB_PARAMS_FOR_STANDARD_MODE.copy()
        effective_n_bootstraps = n_bootstraps
        run_optuna = False
    elif forecast_mode == "advanced_xgb":
        model_to_use = XGBRegressor
        effective_n_bootstraps = n_bootstraps
        run_optuna = True
    else:
        raise ValueError(f"Invalid forecast_mode: {forecast_mode}. Choose 'lower_rf', 'standard_xgb', or 'advanced_xgb'.")

    n_test = min(3, len(X) - 1) if len(X) > 1 else 0
    X_train, X_test = X.iloc[:-n_test] if n_test > 0 else X, X.iloc[-n_test:] if n_test > 0 else pd.DataFrame()
    y_train, y_test = y.iloc[:-n_test] if n_test > 0 else y, y.iloc[-n_test:] if n_test > 0 else pd.Series()
    
    if run_optuna:
        if model_to_use != XGBRegressor:
             raise ValueError("Optuna optimization is only configured for XGBoost ('advanced_xgb' mode).")
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 2, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 1),
                "objective": "reg:squarederror",
                "random_state": 42,
                "n_jobs": -1
            }
            model = XGBRegressor(**params)
            if not X_train.empty and not y_train.empty:
                model.fit(X_train, y_train)
                if not X_test.empty and not y_test.empty:
                    preds = model.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    return rmse
            return float('inf')
        study = optuna.create_study(direction="minimize")
        actual_n_trials = n_trials if len(X_train) >= n_test + 1 and n_test > 0 else max(1, n_trials // 5)
        study.optimize(objective, n_trials=actual_n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_params['objective'] = "reg:squarederror"
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        
    final_model = model_to_use(**best_params)
    final_model.fit(X, y)
    
    last_date = catering_df['Month'].max()
    tz = last_date.tz
    future_months = [last_date + pd.DateOffset(months=i+1) for i in range(forecast_horizon)]
    if tz is not None:
        future_months = [date.tz_localize(tz) if date.tz is None else date for date in future_months]
    
    forecast_df = pd.DataFrame({
        'Month': future_months,
        'ForecastedRevenue': [np.nan] * forecast_horizon, 
        'Lower_CI': [np.nan] * forecast_horizon,
        'Upper_CI': [np.nan] * forecast_horizon
    })
    
    main_model_preds = np.zeros(forecast_horizon)
    temp_y_for_pred = y.copy()
    feature_cols_ordered = list(X.columns)

    for step in range(forecast_horizon):
        month = future_months[step]
        current_features = {
            'Year': month.year, 'Month_Num': month.month,
            'sin_month': np.sin(2 * np.pi * month.month / 12),
            'cos_month': np.cos(2 * np.pi * month.month / 12)
        }
        series_with_forecasts = pd.concat([temp_y_for_pred, pd.Series(main_model_preds[:step])], ignore_index=True)
        for lag_val in [1, 2, 3, 6, 12]:
            lag_col_name = f'lag_{lag_val}'
            if lag_col_name in feature_cols_ordered:
                 if len(series_with_forecasts) >= lag_val:
                     current_features[lag_col_name] = series_with_forecasts.iloc[-lag_val]
                 else:
                     current_features[lag_col_name] = np.nan
        if 'MA_3' in feature_cols_ordered:
            current_features['MA_3'] = series_with_forecasts.iloc[-3:].mean() if len(series_with_forecasts) >=3 else np.nan
        if 'Vol_3' in feature_cols_ordered:
            current_features['Vol_3'] = series_with_forecasts.iloc[-3:].std() if len(series_with_forecasts) >=3 else np.nan
        if 'Momentum_1' in feature_cols_ordered:
             if len(series_with_forecasts) >= 2:
                 mom_val = (series_with_forecasts.iloc[-1] - series_with_forecasts.iloc[-2]) / series_with_forecasts.iloc[-2] if series_with_forecasts.iloc[-2] != 0 else 0
                 current_features['Momentum_1'] = mom_val
             else: 
                 current_features['Momentum_1'] = np.nan
        
        pred_input_df = pd.DataFrame([current_features])[feature_cols_ordered]
        prediction = final_model.predict(pred_input_df)[0]
        main_model_preds[step] = prediction

    forecast_df['ForecastedRevenue'] = main_model_preds

    block_size = min(12, len(X)) if len(X) > 0 else 1
    if effective_n_bootstraps > 0 and len(X) > 0 and len(X) >= block_size:
        boot_preds_ci = np.zeros((effective_n_bootstraps, forecast_horizon))
        blocks = []
        if len(X) >= block_size:
            for i in range(len(X) - block_size + 1):
                X_block = X.iloc[i:i+block_size]
                y_block = y.iloc[i:i+block_size]
                blocks.append((X_block, y_block))
        
        if blocks:
            k_val = max(1, int(np.ceil(len(X)/block_size)))
            for b in range(effective_n_bootstraps):
                random.seed(b)
                sampled_blocks = random.choices(blocks, k=k_val)
                X_boot = pd.concat([block[0] for block in sampled_blocks]).reset_index(drop=True)
                y_boot = pd.concat([block[1] for block in sampled_blocks]).reset_index(drop=True)
                
                boot_model_params = best_params.copy()
                if 'random_state' in boot_model_params: boot_model_params['random_state'] = b
                if model_to_use == XGBRegressor and 'n_jobs' in boot_model_params: boot_model_params['n_jobs'] = 1
                if model_to_use == RandomForestRegressor and 'n_jobs' in boot_model_params: boot_model_params['n_jobs'] = 1

                boot_model = model_to_use(**boot_model_params)
                boot_model.fit(X_boot, y_boot)
                
                current_boot_preds = np.zeros(forecast_horizon)
                temp_y_boot_for_pred = y_boot.copy()
                for step_b in range(forecast_horizon):
                    month_b = future_months[step_b]
                    current_features_boot = {
                        'Year': month_b.year, 'Month_Num': month_b.month,
                        'sin_month': np.sin(2 * np.pi * month_b.month / 12),
                        'cos_month': np.cos(2 * np.pi * month_b.month / 12)
                    }
                    series_with_boot_forecasts = pd.concat([temp_y_boot_for_pred, pd.Series(current_boot_preds[:step_b])], ignore_index=True)
                    for lag_val in [1, 2, 3, 6, 12]:
                        lag_col_name = f'lag_{lag_val}'
                        if lag_col_name in feature_cols_ordered:
                             if len(series_with_boot_forecasts) >= lag_val:
                                 current_features_boot[lag_col_name] = series_with_boot_forecasts.iloc[-lag_val]
                             else:
                                 current_features_boot[lag_col_name] = np.nan
                    if 'MA_3' in feature_cols_ordered:
                        current_features_boot['MA_3'] = series_with_boot_forecasts.iloc[-3:].mean() if len(series_with_boot_forecasts) >=3 else np.nan
                    if 'Vol_3' in feature_cols_ordered:
                        current_features_boot['Vol_3'] = series_with_boot_forecasts.iloc[-3:].std() if len(series_with_boot_forecasts) >=3 else np.nan
                    if 'Momentum_1' in feature_cols_ordered:
                         if len(series_with_boot_forecasts) >= 2:
                             mom_val_boot = (series_with_boot_forecasts.iloc[-1] - series_with_boot_forecasts.iloc[-2]) / series_with_boot_forecasts.iloc[-2] if series_with_boot_forecasts.iloc[-2] != 0 else 0
                             current_features_boot['Momentum_1'] = mom_val_boot
                         else: 
                             current_features_boot['Momentum_1'] = np.nan
                    
                    pred_input_df_boot = pd.DataFrame([current_features_boot])[feature_cols_ordered]
                    prediction_boot = boot_model.predict(pred_input_df_boot)[0]
                    current_boot_preds[step_b] = prediction_boot
                boot_preds_ci[b, :] = current_boot_preds
            
            forecast_df['Lower_CI'] = np.percentile(boot_preds_ci, 5, axis=0)
            forecast_df['Upper_CI'] = np.percentile(boot_preds_ci, 95, axis=0)
            
    return forecast_df 
