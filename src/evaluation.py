import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import config
import os


def calculate_rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


def calculate_mae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return mae


def calculate_mape(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape


def evaluate_model(model_name, y_true, y_pred, training_time):
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    results = {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'training_time': training_time
    }
    
    return results


def compare_models(results_list):
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values('rmse')
    
    print("\n" + "="*70)
    print("COMPARACION DE MODELOS")
    print("="*70)
    print(df_results.to_string(index=False))
    print("="*70)
    
    best_model = df_results.iloc[0]['model']
    best_rmse = df_results.iloc[0]['rmse']
    
    print(f"\nMejor modelo: {best_model}")
    print(f"RMSE: {best_rmse:.4f}")
    
    return df_results


def plot_rmse_comparison(results_df):
    os.makedirs(config.REPORTS_PATH + 'figures/', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['model'], results_df['rmse'], color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.xlabel('Modelo')
    plt.ylabel('RMSE')
    plt.title('Comparacion de RMSE por Modelo')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filepath = config.REPORTS_PATH + 'figures/rmse_comparison.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nGrafico guardado: {filepath}")
    plt.close()


def plot_training_time(results_df):
    os.makedirs(config.REPORTS_PATH + 'figures/', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['model'], results_df['training_time'], color=['#9b59b6', '#f39c12', '#1abc9c'])
    plt.xlabel('Modelo')
    plt.ylabel('Tiempo (segundos)')
    plt.title('Tiempo de Entrenamiento por Modelo')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filepath = config.REPORTS_PATH + 'figures/training_time.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Grafico guardado: {filepath}")
    plt.close()


def plot_predictions_vs_real(y_test, predictions_dict):
    os.makedirs(config.REPORTS_PATH + 'figures/', exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    x = range(len(y_test))
    plt.plot(x, y_test.values, label='Real', color='black', linewidth=2, marker='o', markersize=4)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
        plt.plot(x, predictions, label=model_name, alpha=0.7, linewidth=1.5, color=colors[idx])
    
    plt.xlabel('Observacion')
    plt.ylabel('Demanda')
    plt.title('Predicciones vs Valores Reales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = config.REPORTS_PATH + 'figures/predictions_vs_real.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Grafico guardado: {filepath}")
    plt.close()


def calculate_stock_optimo(demanda_predicha, stock_seguridad=50):
    stock_optimo = demanda_predicha + stock_seguridad
    return stock_optimo


def calculate_roi(beneficio, costo_implementacion):
    roi = ((beneficio - costo_implementacion) / costo_implementacion) * 100
    return roi


def save_results(results_df, filename='resultados_comparacion.csv'):
    os.makedirs(config.REPORTS_PATH, exist_ok=True)
    filepath = config.REPORTS_PATH + filename
    results_df.to_csv(filepath, index=False)
    print(f"\nResultados guardados: {filepath}")


def generate_full_report(results_list, y_test, predictions_dict):
    print("\n" + "="*70)
    print("GENERANDO REPORTE COMPLETO")
    print("="*70)
    
    results_df = compare_models(results_list)
    
    plot_rmse_comparison(results_df)
    plot_training_time(results_df)
    plot_predictions_vs_real(y_test, predictions_dict)
    
    save_results(results_df)
    
    print("\n" + "="*70)
    print("REPORTE COMPLETADO")
    print("="*70)
    
    return results_df


if __name__ == "__main__":
    print("Modulo de evaluacion cargado correctamente")