#configuracion del proyecto

# Rutas
DATA_RAW_PATH = 'data/raw/'
DATA_PROCESSED_PATH = 'data/processed/'
MODELS_PATH = 'models/'
REPORTS_PATH = 'reports/'

# Parámetros del dataset
TRAIN_SIZE = 0.8
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Parámetros Random Forest
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}

# Parámetros XGBoost
XGB_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'random_state': 42
}

# Parámetros Ridge Regression
RIDGE_PARAMS = {
    'alpha': 1.0,
    'random_state': 42
}