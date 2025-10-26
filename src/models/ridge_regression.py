from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import pickle
import time
import config
import os


def train_ridge(X_train, y_train, params=None):
    print("\n" + "="*60)
    print(" ENTRENANDO RIDGE REGRESSION")
    print("="*60)
    
    if params is None:
        params = config.RIDGE_PARAMS
    
    print(f" Parámetros:")
    for key, value in params.items():
        print(f"   - {key}: {value}")
    
    model = Ridge(**params)
    
    print(f"\n Entrenando con {len(X_train)} registros...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    print(f" Entrenamiento completado en {training_time:.2f} segundos")
    
    return model, training_time


def evaluate_ridge(model, X_test, y_test):
    print("\n EVALUANDO MODELO...")
    
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = np.mean(np.abs(y_test - y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    results = {
        'model_name': 'Ridge Regression',
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'n_test': len(y_test)
    }
    
    print(f"\n RESULTADOS:")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - MAE:  {mae:.4f}")
    print(f"   - MAPE: {mape:.2f}%")
    
    return results, y_pred


def save_model(model, filename='ridge.pkl'):
    print(f"\n Guardando modelo...")
    
    os.makedirs(config.MODELS_PATH, exist_ok=True)
    filepath = os.path.join(config.MODELS_PATH, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f" Modelo guardado en: {filepath}")


def load_model(filename='ridge.pkl'):
    filepath = os.path.join(config.MODELS_PATH, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Modelo no encontrado: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f" Modelo cargado desde: {filepath}")
    return model


def get_coefficients(model, feature_names):
    coefficients = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_
    })
    
    coefficients['abs_coefficient'] = np.abs(coefficients['coefficient'])
    coefficients = coefficients.sort_values('abs_coefficient', ascending=False)
    
    print("\n COEFICIENTES (Top 10):")
    for idx, row in coefficients.head(10).iterrows():
        print(f"   {row['feature']:20s}: {row['coefficient']:8.4f}")
    
    return coefficients


def predict_demand(model, X_new):
    predictions = model.predict(X_new)
    return predictions


def run_ridge_pipeline(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print(" PIPELINE RIDGE REGRESSION")
    print("="*60)
    
    model, training_time = train_ridge(X_train, y_train)
    results, predictions = evaluate_ridge(model, X_test, y_test)
    results['training_time'] = training_time
    coefficients = get_coefficients(model, X_train.columns.tolist())
    save_model(model)
    
    print("\n" + "="*60)
    print(" PIPELINE COMPLETADO")
    print("="*60)
    
    return {
        'model': model,
        'results': results,
        'predictions': predictions,
        'coefficients': coefficients
    }


if __name__ == "__main__":
    print(" PROBANDO RIDGE REGRESSION")
    print("="*60)
    
    try:
        X_train = pd.read_csv(f'{config.DATA_PROCESSED_PATH}X_train.csv')
        X_test = pd.read_csv(f'{config.DATA_PROCESSED_PATH}X_test.csv')
        y_train = pd.read_csv(f'{config.DATA_PROCESSED_PATH}y_train.csv').squeeze()
        y_test = pd.read_csv(f'{config.DATA_PROCESSED_PATH}y_test.csv').squeeze()
        
        print(f" Datos cargados")
        print(f"   Train: {X_train.shape}")
        print(f"   Test: {X_test.shape}")
        
        output = run_ridge_pipeline(X_train, X_test, y_train, y_test)
        
        print("\n RESUMEN FINAL:")
        print(f"   RMSE: {output['results']['rmse']:.4f}")
        print(f"   Tiempo: {output['results']['training_time']:.2f}s")
        
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print(" Primero ejecuta: python src/data_preparation.py")
    except Exception as e:
        print(f"\n Error inesperado: {e}")