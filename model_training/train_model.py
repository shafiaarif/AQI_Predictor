"""
train_model.py (V5 - MULTI FEATURE SET EXPERIMENT)

Automatically compares 5 feature sets:

    FS50  -> feature_columns_selected_50.json
    FS70  -> feature_columns_selected_70.json
    FS90  -> feature_columns_selected_90.json
    FS110 -> feature_columns_selected_110.json
    FS130 -> feature_columns_selected_130.json

For EACH feature set, trains:

    1. Persistence Baseline
    2. Random Forest
    3. Ridge Regression
    4. XGBoost
    5. CatBoost
    6. Neural Network
    7. CatBoost + Neural Network Ensemble

Targets:

    target_change_24
    target_change_48
    target_change_72

Absolute AQI reconstruction:

    predicted_aqi = current_us_aqi + predicted_change

IMPORTANT:
    Test data is used ONLY for final evaluation.
    Feature selection was already performed using data before
    the final 90-day period.

OUTPUT:

    saved_models/
        fs_50/
        fs_70/
        fs_90/
        fs_110/
        fs_130/

    data/processed/
        multi_feature_set_results.json
        multi_feature_set_results.csv
        best_model.json
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import hopsworks

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import tensorflow as tf
from tensorflow.keras import layers, callbacks

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

FEATURE_SELECTION_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "feature_selection_sets"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models"
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 10

FEATURE_SET_SIZES = [50, 70, 90, 110, 130]

TARGET_COLUMNS = [
    "target_aqi_24",
    "target_aqi_48",
    "target_aqi_72",
]

CHANGE_TARGET_COLUMNS = [
    "target_change_24",
    "target_change_48",
    "target_change_72",
]

CHANGE_TO_ABSOLUTE = dict(
    zip(
        CHANGE_TARGET_COLUMNS,
        TARGET_COLUMNS
    )
)

HORIZON_LABELS = {
    "target_change_24": "24h",
    "target_change_48": "48h",
    "target_change_72": "72h",
}

TEST_DAYS = 60
VAL_DAYS = 30

RANDOM_STATE = 42

CAT_WEIGHT = 0.50
NN_WEIGHT = 0.50


# ============================================================
# UTILITY
# ============================================================

def print_section(title):

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# 1. CONNECT TO HOPSWORKS
# ============================================================

print_section("[1] CONNECTING TO HOPSWORKS")

api_key = os.environ.get("HOPSWORKS_API_KEY")
project_name = os.environ.get("HOPSWORKS_PROJECT")

if not api_key:
    raise EnvironmentError(
        "HOPSWORKS_API_KEY environment variable not set."
    )

if not project_name:
    raise EnvironmentError(
        "HOPSWORKS_PROJECT environment variable not set."
    )

project = hopsworks.login(
    api_key_value=api_key,
    project=project_name
)

fs = project.get_feature_store()

print("Connected to Hopsworks!")
print("Project:", project.name)


# ============================================================
# 2. LOAD FEATURE DATA
# ============================================================

print_section("[2] LOADING FEATURE DATA")

feature_group = fs.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION
)

df = feature_group.read()

df["time"] = pd.to_datetime(
    df["time"]
)

df = (
    df
    .sort_values("time")
    .reset_index(drop=True)
)

print("Loaded rows   :", len(df))
print("Loaded columns:", len(df.columns))

print(
    "Date range    :",
    df["time"].min(),
    "→",
    df["time"].max()
)


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

print_section("[3] VALIDATING DATA")

required_columns = (
    TARGET_COLUMNS
    + CHANGE_TARGET_COLUMNS
    + ["time", "us_aqi"]
)

missing_columns = [
    c
    for c in required_columns
    if c not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("All required columns are present.")


# ============================================================
# 4. TIME-BASED SPLIT
# ============================================================

print_section(
    "[4] TIME-BASED TRAIN / VALIDATION / TEST SPLIT"
)

test_cutoff = (
    df["time"].max()
    - pd.Timedelta(days=TEST_DAYS)
)

val_cutoff = (
    test_cutoff
    - pd.Timedelta(days=VAL_DAYS)
)

train_df = (
    df[
        df["time"] < val_cutoff
    ]
    .reset_index(drop=True)
)

val_df = (
    df[
        (df["time"] >= val_cutoff)
        &
        (df["time"] < test_cutoff)
    ]
    .reset_index(drop=True)
)

test_df = (
    df[
        df["time"] >= test_cutoff
    ]
    .reset_index(drop=True)
)


print(
    f"Train rows: {len(train_df):>6}"
)

print(
    f"Val rows  : {len(val_df):>6}"
)

print(
    f"Test rows : {len(test_df):>6}"
)

print()

print(
    "Train:",
    train_df["time"].min(),
    "→",
    train_df["time"].max()
)

print(
    "Val  :",
    val_df["time"].min(),
    "→",
    val_df["time"].max()
)

print(
    "Test :",
    test_df["time"].min(),
    "→",
    test_df["time"].max()
)


if len(train_df) == 0:
    raise ValueError("Training set is empty.")

if len(val_df) < 72:
    raise ValueError(
        "Validation set has fewer than 72 rows."
    )

if len(test_df) < 72:
    raise ValueError(
        "Test set has fewer than 72 rows."
    )


# ============================================================
# 5. EVALUATION FUNCTION
# ============================================================

def evaluate_reconstructed(
    current_aqi,
    predicted_change,
    actual_absolute_aqi
):

    predicted_absolute_aqi = (
        np.asarray(current_aqi)
        +
        np.asarray(predicted_change)
    )

    actual_absolute_aqi = np.asarray(
        actual_absolute_aqi
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_absolute_aqi,
            predicted_absolute_aqi
        )
    )

    mae = mean_absolute_error(
        actual_absolute_aqi,
        predicted_absolute_aqi
    )

    r2 = r2_score(
        actual_absolute_aqi,
        predicted_absolute_aqi
    )

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


# ============================================================
# 6. MODEL SUMMARY
# ============================================================

def summarize_model(
    model_name,
    feature_set,
    per_horizon
):

    rmses = [
        metrics["rmse"]
        for metrics in per_horizon.values()
    ]

    maes = [
        metrics["mae"]
        for metrics in per_horizon.values()
    ]

    r2s = [
        metrics["r2"]
        for metrics in per_horizon.values()
    ]

    mean_rmse = float(
        np.mean(rmses)
    )

    mean_mae = float(
        np.mean(maes)
    )

    mean_r2 = float(
        np.mean(r2s)
    )

    print()
    print(
        f"{feature_set:<6} | "
        f"{model_name}"
    )

    for horizon, metrics in per_horizon.items():

        print(
            f"       {HORIZON_LABELS[horizon]:>4} "
            f"RMSE={metrics['rmse']:7.2f} "
            f"MAE={metrics['mae']:7.2f} "
            f"R²={metrics['r2']:.4f}"
        )

    print(
        f"       AVG  "
        f"RMSE={mean_rmse:7.2f} "
        f"MAE={mean_mae:7.2f} "
        f"R²={mean_r2:.4f}"
    )

    return {
        "feature_set": feature_set,
        "model": model_name,
        "per_horizon": per_horizon,
        "mean_rmse": mean_rmse,
        "mean_mae": mean_mae,
        "mean_r2": mean_r2,
    }


# ============================================================
# 7. FEATURE SET LOADING
# ============================================================

def load_feature_set(feature_set_size):

    filename = (
        f"feature_columns_selected_"
        f"{feature_set_size}.json"
    )

    path = os.path.join(
        FEATURE_SELECTION_DIR,
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Feature set not found:\n{path}"
        )

    with open(
        path,
        "r"
    ) as f:

        selected_features = json.load(f)

    excluded = (
        TARGET_COLUMNS
        + CHANGE_TARGET_COLUMNS
        + ["time"]
    )

    valid_features = [
        c
        for c in selected_features
        if c in df.columns
        and c not in excluded
    ]

    missing_features = [
        c
        for c in selected_features
        if c not in df.columns
    ]

    if missing_features:

        print(
            f"WARNING: {len(missing_features)} "
            f"features missing from Hopsworks."
        )

        print(
            "Missing:",
            missing_features[:10]
        )

    if len(valid_features) == 0:

        raise ValueError(
            f"No valid features found for FS{feature_set_size}."
        )

    return valid_features


# ============================================================
# 8. TRAIN ONE FEATURE SET
# ============================================================

def train_feature_set(
    feature_set_size
):

    feature_set_name = (
        f"FS{feature_set_size}"
    )

    print_section(
        f"[FEATURE SET {feature_set_size}]"
    )

    feature_columns = load_feature_set(
        feature_set_size
    )

    print(
        "Feature set:",
        feature_set_name
    )

    print(
        "Number of features:",
        len(feature_columns)
    )

    # --------------------------------------------------------
    # Create X / Y
    # --------------------------------------------------------

    X_train = train_df[
        feature_columns
    ]

    X_val = val_df[
        feature_columns
    ]

    X_test = test_df[
        feature_columns
    ]

    y_train = train_df[
        CHANGE_TARGET_COLUMNS
    ]

    y_val = val_df[
        CHANGE_TARGET_COLUMNS
    ]

    y_test = test_df[
        CHANGE_TARGET_COLUMNS
    ]

    current_aqi_train = (
        train_df["us_aqi"]
    )

    current_aqi_val = (
        val_df["us_aqi"]
    )

    current_aqi_test = (
        test_df["us_aqi"]
    )

    y_train_absolute = train_df[
        TARGET_COLUMNS
    ]

    y_val_absolute = val_df[
        TARGET_COLUMNS
    ]

    y_test_absolute = test_df[
        TARGET_COLUMNS
    ]

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # Model storage
    # --------------------------------------------------------

    trained_models = {}

    results = []

    feature_model_dir = os.path.join(
        MODEL_DIR,
        feature_set_name.lower()
    )

    os.makedirs(
        feature_model_dir,
        exist_ok=True
    )

    # ========================================================
    # PERSISTENCE BASELINE
    # ========================================================

    print_section(
        f"{feature_set_name} - PERSISTENCE BASELINE"
    )

    baseline_per_horizon = {}

    for change_col, abs_col in CHANGE_TO_ABSOLUTE.items():

        zero_change = np.zeros(
            len(test_df)
        )

        baseline_per_horizon[
            change_col
        ] = evaluate_reconstructed(

            current_aqi_test,

            zero_change,

            y_test_absolute[abs_col]
        )

    baseline_result = summarize_model(
        "Persistence Baseline",
        feature_set_name,
        baseline_per_horizon
    )

    results.append(
        baseline_result
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print_section(
        f"{feature_set_name} - RANDOM FOREST"
    )

    rf_param_dist = {

        "n_estimators": [
            200,
            300,
            500
        ],

        "max_depth": [
            6,
            10,
            15,
            20
        ],

        "min_samples_leaf": [
            1,
            2,
            4,
            8
        ],

        "max_features": [
            "sqrt",
            0.5,
            0.8
        ]
    }

    rf_per_horizon = {}

    trained_models[
        "random_forest"
    ] = {}

    for change_col, abs_col in CHANGE_TO_ABSOLUTE.items():

        print(
            f"\nTraining RF for "
            f"{HORIZON_LABELS[change_col]}..."
        )

        # IMPORTANT:
        # Tune only on TRAIN.
        # Validation remains untouched for this stage.

        tscv = TimeSeriesSplit(
            n_splits=3
        )

        search = RandomizedSearchCV(

            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

            rf_param_dist,

            n_iter=8,

            cv=tscv,

            scoring="neg_root_mean_squared_error",

            random_state=RANDOM_STATE,

            n_jobs=-1
        )

        search.fit(
            X_train,
            y_train[change_col]
        )

        best_model = search.best_estimator_

        # Final RF is refit on TRAIN + VAL
        # using selected hyperparameters.

        X_trainval = pd.concat(
            [
                X_train,
                X_val
            ],
            ignore_index=True
        )

        y_trainval_h = pd.concat(
            [
                y_train[change_col],
                y_val[change_col]
            ],
            ignore_index=True
        )

        final_rf = RandomForestRegressor(
            **search.best_params_,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        final_rf.fit(
            X_trainval,
            y_trainval_h
        )

        pred_change = final_rf.predict(
            X_test
        )

        rf_per_horizon[
            change_col
        ] = evaluate_reconstructed(

            current_aqi_test,

            pred_change,

            y_test_absolute[abs_col]
        )

        trained_models[
            "random_forest"
        ][change_col] = final_rf

        print(
            "Best params:",
            search.best_params_
        )

    rf_result = summarize_model(
        "Random Forest",
        feature_set_name,
        rf_per_horizon
    )

    results.append(
        rf_result
    )

    # ========================================================
    # RIDGE
    # ========================================================

    print_section(
        f"{feature_set_name} - RIDGE"
    )

    ridge_per_horizon = {}

    trained_models[
        "ridge"
    ] = {}

    X_trainval_scaled = np.vstack(
        [
            X_train_scaled,
            X_val_scaled
        ]
    )

    for change_col, abs_col in CHANGE_TO_ABSOLUTE.items():

        print(
            f"\nTraining Ridge for "
            f"{HORIZON_LABELS[change_col]}..."
        )

        tscv = TimeSeriesSplit(
            n_splits=4
        )

        ridge = RidgeCV(
            alphas=[
                0.1,
                1.0,
                5.0,
                10.0,
                50.0,
                100.0
            ],
            cv=tscv
        )

        ridge.fit(
            X_trainval_scaled,
            pd.concat(
                [
                    y_train[change_col],
                    y_val[change_col]
                ],
                ignore_index=True
            )
        )

        pred_change = ridge.predict(
            X_test_scaled
        )

        ridge_per_horizon[
            change_col
        ] = evaluate_reconstructed(

            current_aqi_test,

            pred_change,

            y_test_absolute[abs_col]
        )

        trained_models[
            "ridge"
        ][change_col] = ridge

        print(
            "Best alpha:",
            ridge.alpha_
        )

    ridge_result = summarize_model(
        "Ridge Regression",
        feature_set_name,
        ridge_per_horizon
    )

    results.append(
        ridge_result
    )

    # ========================================================
    # XGBOOST
    # ========================================================

    print_section(
        f"{feature_set_name} - XGBOOST"
    )

    xgb_param_grid = [

        {
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },

        {
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },

        {
            "max_depth": 8,
            "learning_rate": 0.02,
            "subsample": 0.7,
            "colsample_bytree": 0.7
        }
    ]

    xgb_per_horizon = {}

    trained_models[
        "xgboost"
    ] = {}

    for change_col, abs_col in CHANGE_TO_ABSOLUTE.items():

        print(
            f"\nTraining XGBoost for "
            f"{HORIZON_LABELS[change_col]}..."
        )

        best_val_rmse = np.inf
        best_params = None
        best_iteration = 1000

        # ----------------------------------------------------
        # Tune on TRAIN -> evaluate on VAL
        # ----------------------------------------------------

        for params in xgb_param_grid:

            model = XGBRegressor(

                n_estimators=3000,

                max_depth=params[
                    "max_depth"
                ],

                learning_rate=params[
                    "learning_rate"
                ],

                subsample=params[
                    "subsample"
                ],

                colsample_bytree=params[
                    "colsample_bytree"
                ],

                reg_lambda=1.0,

                objective="reg:squarederror",

                eval_metric="rmse",

                early_stopping_rounds=50,

                random_state=RANDOM_STATE,

                n_jobs=-1
            )

            model.fit(

                X_train,

                y_train[change_col],

                eval_set=[
                    (
                        X_val,
                        y_val[change_col]
                    )
                ],

                verbose=False
            )

            val_pred = model.predict(
                X_val
            )

            val_metrics = (
                evaluate_reconstructed(

                    current_aqi_val,

                    val_pred,

                    y_val_absolute[abs_col]
                )
            )

            if val_metrics["rmse"] < best_val_rmse:

                best_val_rmse = (
                    val_metrics["rmse"]
                )

                best_params = params.copy()

                if model.best_iteration is not None:

                    best_iteration = (
                        model.best_iteration + 1
                    )

        print(
            "Best params:",
            best_params
        )

        print(
            "Best validation RMSE:",
            f"{best_val_rmse:.4f}"
        )

        print(
            "Best iteration:",
            best_iteration
        )

        # ----------------------------------------------------
        # Refit TRAIN + VAL
        # ----------------------------------------------------

        X_trainval = pd.concat(
            [
                X_train,
                X_val
            ],
            ignore_index=True
        )

        y_trainval_h = pd.concat(
            [
                y_train[change_col],
                y_val[change_col]
            ],
            ignore_index=True
        )

        final_xgb = XGBRegressor(

            n_estimators=best_iteration,

            max_depth=best_params[
                "max_depth"
            ],

            learning_rate=best_params[
                "learning_rate"
            ],

            subsample=best_params[
                "subsample"
            ],

            colsample_bytree=best_params[
                "colsample_bytree"
            ],

            reg_lambda=1.0,

            objective="reg:squarederror",

            eval_metric="rmse",

            random_state=RANDOM_STATE,

            n_jobs=-1
        )

        final_xgb.fit(
            X_trainval,
            y_trainval_h,
            verbose=False
        )

        pred_change = final_xgb.predict(
            X_test
        )

        xgb_per_horizon[
            change_col
        ] = evaluate_reconstructed(

            current_aqi_test,

            pred_change,

            y_test_absolute[abs_col]
        )

        trained_models[
            "xgboost"
        ][change_col] = final_xgb

    xgb_result = summarize_model(
        "XGBoost",
        feature_set_name,
        xgb_per_horizon
    )

    results.append(
        xgb_result
    )

    # ========================================================
    # CATBOOST
    # ========================================================

    print_section(
        f"{feature_set_name} - CATBOOST"
    )

    cat_param_grid = [

        {
            "depth": 4,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3
        },

        {
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 5
        },

        {
            "depth": 8,
            "learning_rate": 0.02,
            "l2_leaf_reg": 7
        }
    ]

    cat_per_horizon = {}

    trained_models[
        "catboost"
    ] = {}

    for change_col, abs_col in CHANGE_TO_ABSOLUTE.items():

        print(
            f"\nTraining CatBoost for "
            f"{HORIZON_LABELS[change_col]}..."
        )

        best_val_rmse = np.inf
        best_params = None
        best_iteration = 1000

        # ----------------------------------------------------
        # Tune using TRAIN -> VAL
        # ----------------------------------------------------

        for params in cat_param_grid:

            model = CatBoostRegressor(

                iterations=3000,

                depth=params[
                    "depth"
                ],

                learning_rate=params[
                    "learning_rate"
                ],

                l2_leaf_reg=params[
                    "l2_leaf_reg"
                ],

                loss_function="RMSE",

                random_seed=RANDOM_STATE,

                early_stopping_rounds=50,

                verbose=False
            )

            model.fit(

                X_train,

                y_train[change_col],

                eval_set=(
                    X_val,
                    y_val[change_col]
                ),

                verbose=False
            )

            val_pred = model.predict(
                X_val
            )

            val_metrics = (
                evaluate_reconstructed(

                    current_aqi_val,

                    val_pred,

                    y_val_absolute[abs_col]
                )
            )

            if val_metrics["rmse"] < best_val_rmse:

                best_val_rmse = (
                    val_metrics["rmse"]
                )

                best_params = params.copy()

                iteration = (
                    model.get_best_iteration()
                )

                if iteration is not None and iteration > 0:

                    best_iteration = (
                        iteration + 1
                    )

        print(
            "Best params:",
            best_params
        )

        print(
            "Best validation RMSE:",
            f"{best_val_rmse:.4f}"
        )

        print(
            "Best iterations:",
            best_iteration
        )

        # ----------------------------------------------------
        # Refit TRAIN + VAL
        # ----------------------------------------------------

        X_trainval = pd.concat(
            [
                X_train,
                X_val
            ],
            ignore_index=True
        )

        y_trainval_h = pd.concat(
            [
                y_train[change_col],
                y_val[change_col]
            ],
            ignore_index=True
        )

        final_cat = CatBoostRegressor(

            iterations=best_iteration,

            depth=best_params[
                "depth"
            ],

            learning_rate=best_params[
                "learning_rate"
            ],

            l2_leaf_reg=best_params[
                "l2_leaf_reg"
            ],

            loss_function="RMSE",

            random_seed=RANDOM_STATE,

            verbose=False
        )

        final_cat.fit(
            X_trainval,
            y_trainval_h,
            verbose=False
        )

        pred_change = final_cat.predict(
            X_test
        )

        cat_per_horizon[
            change_col
        ] = evaluate_reconstructed(

            current_aqi_test,

            pred_change,

            y_test_absolute[abs_col]
        )

        trained_models[
            "catboost"
        ][change_col] = final_cat

    cat_result = summarize_model(
        "CatBoost",
        feature_set_name,
        cat_per_horizon
    )

    results.append(
        cat_result
    )

    # ========================================================
    # NEURAL NETWORK
    # ========================================================

    print_section(
        f"{feature_set_name} - NEURAL NETWORK"
    )

    tf.random.set_seed(
        RANDOM_STATE
    )

    # Use TRAIN + VAL for final NN training,
    # but keep a validation split from the TRAIN portion
    # for early stopping.

    X_nn_train = X_train_scaled
    X_nn_val = X_val_scaled

    nn_model = tf.keras.Sequential([

        layers.Input(
            shape=(
                X_train_scaled.shape[1],
            )
        ),

        layers.Dense(
            256,
            activation="relu"
        ),

        layers.Dropout(0.20),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.20),

        layers.Dense(
            64,
            activation="relu"
        ),

        layers.Dense(
            len(CHANGE_TARGET_COLUMNS),
            activation="linear"
        )
    ])

    nn_model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),

        loss="mse",

        metrics=["mae"]
    )

    early_stop = callbacks.EarlyStopping(

        monitor="val_loss",

        patience=15,

        restore_best_weights=True
    )

    history = nn_model.fit(

        X_nn_train,

        y_train.to_numpy(),

        validation_data=(

            X_nn_val,

            y_val.to_numpy()
        ),

        epochs=300,

        batch_size=64,

        callbacks=[
            early_stop
        ],

        verbose=0
    )

    print(
        "Stopped after:",
        len(
            history.history["loss"]
        ),
        "epochs"
    )

    nn_pred_change = nn_model.predict(
        X_test_scaled,
        verbose=0
    )

    nn_per_horizon = {}

    for i, (
        change_col,
        abs_col
    ) in enumerate(
        CHANGE_TO_ABSOLUTE.items()
    ):

        nn_per_horizon[
            change_col
        ] = evaluate_reconstructed(

            current_aqi_test,

            nn_pred_change[:, i],

            y_test_absolute[abs_col]
        )

    nn_result = summarize_model(
        "Neural Network",
        feature_set_name,
        nn_per_horizon
    )

    results.append(
        nn_result
    )

    # ========================================================
    # ENSEMBLE
    # ========================================================

    print_section(
        f"{feature_set_name} - CATBOOST + NN ENSEMBLE"
    )

    print(
        f"CatBoost weight = {CAT_WEIGHT:.2f}"
    )

    print(
        f"NN weight       = {NN_WEIGHT:.2f}"
    )

    ensemble_per_horizon = {}

    ensemble_predictions = {}

    for i, (
        change_col,
        abs_col
    ) in enumerate(
        CHANGE_TO_ABSOLUTE.items()
    ):

        cat_pred_change = (
            trained_models[
                "catboost"
            ][change_col].predict(
                X_test
            )
        )

        nn_pred_change_horizon = (
            nn_pred_change[:, i]
        )

        ensemble_pred_change = (

            CAT_WEIGHT
            *
            cat_pred_change

            +

            NN_WEIGHT
            *
            nn_pred_change_horizon
        )

        ensemble_predictions[
            change_col
        ] = ensemble_pred_change

        ensemble_per_horizon[
            change_col
        ] = evaluate_reconstructed(

            current_aqi_test,

            ensemble_pred_change,

            y_test_absolute[abs_col]
        )

    ensemble_result = summarize_model(
        "Ensemble (CatBoost + NN)",
        feature_set_name,
        ensemble_per_horizon
    )

    results.append(
        ensemble_result
    )

    # ========================================================
    # SAVE MODELS
    # ========================================================

    print_section(
        f"{feature_set_name} - SAVING MODELS"
    )

    model_dirs = {

        "random_forest":
            "random_forest",

        "ridge":
            "ridge",

        "xgboost":
            "xgboost",

        "catboost":
            "catboost"
    }

    for model_key, models in trained_models.items():

        model_subdir = os.path.join(
            feature_model_dir,
            model_dirs[model_key]
        )

        os.makedirs(
            model_subdir,
            exist_ok=True
        )

        for change_col, model in models.items():

            abs_col = CHANGE_TO_ABSOLUTE[
                change_col
            ]

            path = os.path.join(
                model_subdir,
                f"{abs_col}.pkl"
            )

            joblib.dump(
                model,
                path
            )

        if model_key == "ridge":

            joblib.dump(

                scaler,

                os.path.join(
                    model_subdir,
                    "scaler.pkl"
                )
            )

    # --------------------------------------------------------
    # Save NN
    # --------------------------------------------------------

    nn_dir = os.path.join(
        feature_model_dir,
        "neural_network"
    )

    os.makedirs(
        nn_dir,
        exist_ok=True
    )

    nn_model.save(
        os.path.join(
            nn_dir,
            "model.keras"
        )
    )

    joblib.dump(
        scaler,
        os.path.join(
            nn_dir,
            "scaler.pkl"
        )
    )

    # --------------------------------------------------------
    # Save feature list
    # --------------------------------------------------------

    with open(
        os.path.join(
            feature_model_dir,
            "feature_columns.json"
        ),
        "w"
    ) as f:

        json.dump(
            feature_columns,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Save ensemble config
    # --------------------------------------------------------

    ensemble_config = {

        "feature_set":
            feature_set_name,

        "type":
            "weighted_average",

        "models": [

            {
                "name":
                    "catboost",

                "weight":
                    CAT_WEIGHT
            },

            {
                "name":
                    "neural_network",

                "weight":
                    NN_WEIGHT
            }
        ],

        "prediction_mode":
            "change",

        "reconstruction":
            "predicted_aqi = "
            "current_us_aqi + "
            "weighted_average_predicted_change"
    }

    with open(
        os.path.join(
            feature_model_dir,
            "ensemble_config.json"
        ),
        "w"
    ) as f:

        json.dump(
            ensemble_config,
            f,
            indent=2
        )

    print(
        "Models saved to:",
        feature_model_dir
    )

    return results


# ============================================================
# 9. RUN ALL FIVE FEATURE SETS
# ============================================================

print_section(
    "[5] RUNNING FIVE FEATURE SET EXPERIMENT"
)

print(
    "Feature sets:",
    FEATURE_SET_SIZES
)

print(
    "Models per feature set:"
)

print(
    "  - Persistence"
)

print(
    "  - Random Forest"
)

print(
    "  - Ridge"
)

print(
    "  - XGBoost"
)

print(
    "  - CatBoost"
)

print(
    "  - Neural Network"
)

print(
    "  - CatBoost + NN Ensemble"
)


all_results = []

for feature_set_size in FEATURE_SET_SIZES:

    results = train_feature_set(
        feature_set_size
    )

    all_results.extend(
        results
    )


# ============================================================
# 10. FINAL COMPARISON
# ============================================================

print_section(
    "[6] COMPLETE MODEL COMPARISON"
)

print()

print(
    f"{'Feature':<10}"
    f"{'Model':<32}"
    f"{'RMSE':>10}"
    f"{'MAE':>10}"
    f"{'R²':>10}"
)

print("-" * 75)

for result in all_results:

    print(
        f"{result['feature_set']:<10}"
        f"{result['model']:<32}"
        f"{result['mean_rmse']:>10.2f}"
        f"{result['mean_mae']:>10.2f}"
        f"{result['mean_r2']:>10.4f}"
    )


# ============================================================
# 11. BEST OVERALL MODEL
# ============================================================

best_model = min(
    all_results,
    key=lambda x: x["mean_rmse"]
)

print_section(
    "[7] BEST OVERALL MODEL"
)

print(
    "Best feature set:",
    best_model["feature_set"]
)

print(
    "Best model:",
    best_model["model"]
)

print(
    "Average RMSE:",
    f"{best_model['mean_rmse']:.4f}"
)

print(
    "Average MAE:",
    f"{best_model['mean_mae']:.4f}"
)

print(
    "Average R²:",
    f"{best_model['mean_r2']:.4f}"
)

print()

for horizon, metrics in best_model[
    "per_horizon"
].items():

    print(
        f"{HORIZON_LABELS[horizon]:>4} "
        f"RMSE={metrics['rmse']:.4f} "
        f"MAE={metrics['mae']:.4f} "
        f"R²={metrics['r2']:.4f}"
    )


# ============================================================
# 12. BEST MODEL FOR EACH FEATURE SET
# ============================================================

print_section(
    "[8] BEST MODEL FOR EACH FEATURE SET"
)

best_by_feature_set = {}

for feature_set_size in FEATURE_SET_SIZES:

    feature_set_name = (
        f"FS{feature_set_size}"
    )

    candidates = [
        r
        for r in all_results
        if r["feature_set"] == feature_set_name
    ]

    best = min(
        candidates,
        key=lambda x: x["mean_rmse"]
    )

    best_by_feature_set[
        feature_set_name
    ] = best

    print(
        f"{feature_set_name:<8} "
        f"{best['model']:<32} "
        f"RMSE={best['mean_rmse']:.4f} "
        f"MAE={best['mean_mae']:.4f} "
        f"R²={best['mean_r2']:.4f}"
    )


# ============================================================
# 13. BEST MODEL BY HORIZON
# ============================================================

print_section(
    "[9] BEST MODEL BY FORECAST HORIZON"
)

for horizon in CHANGE_TARGET_COLUMNS:

    horizon_candidates = []

    for result in all_results:

        metrics = result[
            "per_horizon"
        ][horizon]

        horizon_candidates.append(
            {
                "feature_set":
                    result["feature_set"],

                "model":
                    result["model"],

                "rmse":
                    metrics["rmse"],

                "mae":
                    metrics["mae"],

                "r2":
                    metrics["r2"]
            }
        )

    best_horizon = min(
        horizon_candidates,
        key=lambda x: x["rmse"]
    )

    print(
        f"{HORIZON_LABELS[horizon]:>4} -> "
        f"{best_horizon['feature_set']} + "
        f"{best_horizon['model']} | "
        f"RMSE={best_horizon['rmse']:.4f} | "
        f"MAE={best_horizon['mae']:.4f} | "
        f"R²={best_horizon['r2']:.4f}"
    )


# ============================================================
# 14. SAVE RESULTS JSON
# ============================================================

print_section(
    "[10] SAVING RESULTS"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

results_json = []

for result in all_results:

    result_copy = {

        "feature_set":
            result["feature_set"],

        "model":
            result["model"],

        "mean_rmse":
            float(result["mean_rmse"]),

        "mean_mae":
            float(result["mean_mae"]),

        "mean_r2":
            float(result["mean_r2"]),

        "per_horizon": {}
    }

    for horizon, metrics in result[
        "per_horizon"
    ].items():

        result_copy[
            "per_horizon"
        ][horizon] = {

            "rmse":
                float(metrics["rmse"]),

            "mae":
                float(metrics["mae"]),

            "r2":
                float(metrics["r2"])
        }

    results_json.append(
        result_copy
    )


results_json_path = os.path.join(
    RESULTS_DIR,
    "multi_feature_set_results.json"
)

with open(
    results_json_path,
    "w"
) as f:

    json.dump(
        results_json,
        f,
        indent=2
    )

print(
    "Saved:",
    results_json_path
)


# ============================================================
# 15. SAVE RESULTS CSV
# ============================================================

csv_rows = []

for result in all_results:

    row = {

        "feature_set":
            result["feature_set"],

        "model":
            result["model"],

        "mean_rmse":
            result["mean_rmse"],

        "mean_mae":
            result["mean_mae"],

        "mean_r2":
            result["mean_r2"],
    }

    for horizon, metrics in result[
        "per_horizon"
    ].items():

        label = HORIZON_LABELS[
            horizon
        ]

        row[
            f"rmse_{label}"
        ] = metrics["rmse"]

        row[
            f"mae_{label}"
        ] = metrics["mae"]

        row[
            f"r2_{label}"
        ] = metrics["r2"]

    csv_rows.append(
        row
    )


results_csv_path = os.path.join(
    RESULTS_DIR,
    "multi_feature_set_results.csv"
)

results_df = pd.DataFrame(
    csv_rows
)

results_df = results_df.sort_values(
    "mean_rmse"
)

results_df.to_csv(
    results_csv_path,
    index=False
)

print(
    "Saved:",
    results_csv_path
)


# ============================================================
# 16. SAVE BEST MODEL INFORMATION
# ============================================================

best_model_path = os.path.join(
    RESULTS_DIR,
    "best_model.json"
)

best_model_info = {

    "feature_set":
        best_model["feature_set"],

    "model":
        best_model["model"],

    "mean_rmse":
        float(best_model["mean_rmse"]),

    "mean_mae":
        float(best_model["mean_mae"]),

    "mean_r2":
        float(best_model["mean_r2"]),

    "per_horizon":
        best_model["per_horizon"],

    "test_period": {

        "start":
            str(test_df["time"].min()),

        "end":
            str(test_df["time"].max())
    },

    "feature_selection":

        "Leakage-safe selection using data before "
        "the final 90-day period.",

    "prediction_mode":
        "change",

    "reconstruction":
        "predicted_aqi = "
        "current_us_aqi + predicted_change"
}

with open(
    best_model_path,
    "w"
) as f:

    json.dump(
        best_model_info,
        f,
        indent=2
    )

print(
    "Saved:",
    best_model_path
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print_section(
    "[11] TRAINING COMPLETE"
)

print(
    "Five feature sets evaluated:"
)

for feature_set_size in FEATURE_SET_SIZES:

    feature_set_name = (
        f"FS{feature_set_size}"
    )

    best = best_by_feature_set[
        feature_set_name
    ]

    print(
        f"  {feature_set_name:<8} "
        f"{best['model']:<32} "
        f"RMSE={best['mean_rmse']:.2f} "
        f"R²={best['mean_r2']:.4f}"
    )

print()

print(
    "=================================================="
)

print(
    "WINNER"
)

print(
    "=================================================="
)

print(
    "Feature set:",
    best_model["feature_set"]
)

print(
    "Model:",
    best_model["model"]
)

print(
    "Average RMSE:",
    f"{best_model['mean_rmse']:.4f}"
)

print(
    "Average MAE:",
    f"{best_model['mean_mae']:.4f}"
)

print(
    "Average R²:",
    f"{best_model['mean_r2']:.4f}"
)

print()

print(
    "Models saved under:"
)

print(
    MODEL_DIR
)

print()

print(
    "Results:"
)

print(
    results_json_path
)

print(
    results_csv_path
)

print(
    best_model_path
)

print()

print(
    "IMPORTANT:"
)

print(
    "The test set was used only for final evaluation."
)

print(
    "All models predict AQI CHANGE."
)

print(
    "Absolute AQI:"
)

print(
    "predicted_aqi = current_us_aqi + predicted_change"
)

print()

print("=" * 75)
print("DONE")
print("=" * 75)