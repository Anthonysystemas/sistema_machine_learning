
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import pickle
import time
import config
import os


def train_xgboost(X_train, y_train, params=None):
   
    print("\n" + "="*60)
    print(" ENTRENANDO XGBOOST")
    print("="*60)
    
    # Usar parámetros del config si no se proporcionan
    if params is None:
        params = config.XGB_PARAMS
    
    print(f"📋 Parámetros:")
    for key, value in params.items():
        print(f"   - {key}: {value}")
    
    model = xgb.XGBRegressor(**params)
    
    print(f"\n⏳ Entrenando con {len(X_train)} registros...")
    start_time = time.time()
    
    model.fit(X_train, y_train, verbose=False)
    
    training_time = time.time() - start_time
    
    print(f"✅ Entrenamiento completado en {training_time:.2f} segundos")
    print(f"   ({training_time/60:.2f} minutos)")
    
    return model, training_time


def evaluate_xgboost(model, X_test, y_test):
   
    print("\n EVALUANDO MODELO...")
    
    # Hacer predicciones
    y_pred = model.predict(X_test)
    
    # Calcular RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Calcular otras métricas útiles
    mae = np.mean(np.abs(y_test - y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    results = {
        'model_name': 'XGBoost',
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


def save_model(model, filename='xgboost.pkl'):
  
    print(f"\n Guardando modelo...")
    
    # Crear directorio si no existe
    os.makedirs(config.MODELS_PATH, exist_ok=True)
    
    filepath = os.path.join(config.MODELS_PATH, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f" Modelo guardado en: {filepath}")


def load_model(filename='xgboost.pkl'):
   
    filepath = os.path.join(config.MODELS_PATH, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Modelo no encontrado: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ Modelo cargado desde: {filepath}")
    return model


def get_feature_importance(model, feature_names):
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    importance = importance.sort_values('importance', ascending=False)
    
    print("\n IMPORTANCIA DE FEATURES (Top 10):")
    for idx, row in importance.head(10).iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")
    
    return importance


def predict_demand(model, X_new):
    
    predictions = model.predict(X_new)
    return predictions


# Pipeline completo
def run_xgboost_pipeline(X_train, X_test, y_train, y_test):
    
    print("\n" + "="*60)
    print("  XGBOOST")
    print("="*60)
    
    # 1. Entrenar
    model, training_time = train_xgboost(X_train, y_train)
    
    # 2. Evaluar
    results, predictions = evaluate_xgboost(model, X_test, y_test)
    results['training_time'] = training_time
    
    # 3. Importancia de features
    importance = get_feature_importance(model, X_train.columns.tolist())
    
    # 4. Guardar modelo
    save_model(model)
    
    print("\n" + "="*60)
    print(" PIPELINE COMPLETADO")
    print("="*60)
    
    return {
        'model': model,
        'results': results,
        'predictions': predictions,
        'feature_importance': importance
    }


if __name__ == "__main__":
    print(" PROBANDO XGBOOST")
    print("="*60)
    
    # Cargar datos procesados
    try:
        X_train = pd.read_csv(f'{config.DATA_PROCESSED_PATH}X_train.csv')
        X_test = pd.read_csv(f'{config.DATA_PROCESSED_PATH}X_test.csv')
        y_train = pd.read_csv(f'{config.DATA_PROCESSED_PATH}y_train.csv').squeeze()
        y_test = pd.read_csv(f'{config.DATA_PROCESSED_PATH}y_test.csv').squeeze()
        
        print(f" Datos cargados")
        print(f"   Train: {X_train.shape}")
        print(f"   Test: {X_test.shape}")
        
        output = run_xgboost_pipeline(X_train, X_test, y_train, y_test)

        print("\n RESUMEN FINAL:")
        print(f"   RMSE: {output['results']['rmse']:.4f}")
        print(f"   Tiempo: {output['results']['training_time']:.2f}s")
        
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print(" Primero ejecuta: python src/data_preparation.py")
    except Exception as e:
        print(f"\n Error inesperado: {e}")