import pandas as pd
from src.data_preparation import load_and_preprocess_data
from src.models.random_forest import run_random_forest_pipeline
from src.models.xgboost_model import run_xgboost_pipeline
from src.models.ridge_regression import run_ridge_pipeline
from src.evaluation import generate_full_report, calculate_stock_optimo, calculate_roi
import config


def main():
    print("\n" + "="*70)
    print("SISTEMA DE PREDICCION DE DEMANDA CON MACHINE LEARNING")
    print("="*70)
    
    print("\n[PASO 1/5] Cargando y preparando datos...")
    try:
        X_train = pd.read_csv(f'{config.DATA_PROCESSED_PATH}X_train.csv')
        X_test = pd.read_csv(f'{config.DATA_PROCESSED_PATH}X_test.csv')
        y_train = pd.read_csv(f'{config.DATA_PROCESSED_PATH}y_train.csv').squeeze()
        y_test = pd.read_csv(f'{config.DATA_PROCESSED_PATH}y_test.csv').squeeze()
        print(f"Datos cargados exitosamente")
        print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    except FileNotFoundError:
        print("Datos no encontrados. Ejecutando preprocesamiento...")
        file_path = input("Ingresa la ruta del archivo de datos: ")
        target_column = input("Ingresa el nombre de la columna objetivo: ")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, target_column)
    
    print("\n[PASO 2/5] Entrenando Random Forest...")
    rf_output = run_random_forest_pipeline(X_train, X_test, y_train, y_test)
    
    print("\n[PASO 3/5] Entrenando XGBoost...")
    xgb_output = run_xgboost_pipeline(X_train, X_test, y_train, y_test)
    
    print("\n[PASO 4/5] Entrenando Ridge Regression...")
    ridge_output = run_ridge_pipeline(X_train, X_test, y_train, y_test)
    
    print("\n[PASO 5/5] Generando reporte comparativo...")
    
    results_list = [
        rf_output['results'],
        xgb_output['results'],
        ridge_output['results']
    ]
    
    predictions_dict = {
        'Random Forest': rf_output['predictions'],
        'XGBoost': xgb_output['predictions'],
        'Ridge Regression': ridge_output['predictions']
    }
    
    results_df = generate_full_report(results_list, y_test, predictions_dict)
    
    best_model_name = results_df.iloc[0]['model']
    best_rmse = results_df.iloc[0]['rmse']
    
    if best_model_name == 'Random Forest':
        best_predictions = rf_output['predictions']
    elif best_model_name == 'XGBoost':
        best_predictions = xgb_output['predictions']
    else:
        best_predictions = ridge_output['predictions']
    
    print("\n" + "="*70)
    print("CALCULO DE INDICADORES")
    print("="*70)
    
    demanda_promedio = best_predictions.mean()
    stock_optimo = calculate_stock_optimo(demanda_promedio, stock_seguridad=50)
    
    print(f"\nDemanda promedio predicha: {demanda_promedio:.2f} unidades")
    print(f"Stock optimo recomendado: {stock_optimo:.2f} unidades")
    
    costo_implementacion = 5000
    beneficio_estimado = 9000
    roi = calculate_roi(beneficio_estimado, costo_implementacion)
    
    print(f"\nCosto de implementacion: S/ {costo_implementacion:,.2f}")
    print(f"Beneficio estimado: S/ {beneficio_estimado:,.2f}")
    print(f"ROI: {roi:.2f}%")
    
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"Mejor modelo: {best_model_name}")
    print(f"RMSE: {best_rmse:.4f}")
    print(f"Stock optimo: {stock_optimo:.2f} unidades")
    print(f"ROI: {roi:.2f}%")
    print("="*70)
    
    print("\nGraficos guardados en: reports/figures/")
    print("Resultados guardados en: reports/resultados_comparacion.csv")
    print("\nProceso completado exitosamente")


if __name__ == "__main__":
    main()