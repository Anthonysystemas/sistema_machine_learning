import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config
import os

def load_raw_data(file_path):
    print(f" Cargando datos desde: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Formato no soportado. Usa CSV o Excel (.xlsx, .xls)")
    
    print(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"   Columnas: {list(df.columns)}")
    return df


def clean_data(df):
    print("\n Limpiando datos...")
    
    inicial = len(df)
    
    df = df.drop_duplicates()
    duplicados = inicial - len(df)
    print(f"   ✓ Duplicados eliminados: {duplicados}")
    
    nulos = df.isnull().sum()
    if nulos.sum() > 0:
        print(f"   ⚠ Valores nulos encontrados:")
        for col in nulos[nulos > 0].index:
            print(f"      - {col}: {nulos[col]} ({nulos[col]/len(df)*100:.1f}%)")
    
    df = df.dropna()
    print(f"   ✓ Filas con nulos eliminadas: {inicial - len(df)}")
    
    print(f"✅ Datos limpios: {len(df)} registros restantes")
    return df


def prepare_features(df):
    print("\n Preparando features...")
    
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['año'] = df['fecha'].dt.year
        df['mes'] = df['fecha'].dt.month
        df['semana'] = df['fecha'].dt.isocalendar().week
        df['dia_semana'] = df['fecha'].dt.dayofweek
        print(f"   ✓ Features temporales creadas")
        
        df = df.drop(columns=['fecha'])
    
    print(f" Features preparadas: {df.shape[1]} columnas")
    return df


def normalize_data(X_train, X_test):
    print("\n Normalizando datos...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(f" Datos normalizados")
    return X_train_scaled, X_test_scaled, scaler


def split_data(df, target_column):
    print(f"\n Dividiendo datos (train: {config.TRAIN_SIZE*100}%, test: {config.TEST_SIZE*100}%)...")
    
    if target_column not in df.columns:
        raise ValueError(f"Columna '{target_column}' no encontrada. Columnas disponibles: {list(df.columns)}")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        shuffle=True
    )
    
    print(f" División completada:")
    print(f"   - Train: {len(X_train)} registros")
    print(f"   - Test: {len(X_test)} registros")
    
    return X_train, X_test, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test):
    print("\n Guardando datos procesados...")
    
    os.makedirs(config.DATA_PROCESSED_PATH, exist_ok=True)
    
    X_train.to_csv(f'{config.DATA_PROCESSED_PATH}X_train.csv', index=False)
    X_test.to_csv(f'{config.DATA_PROCESSED_PATH}X_test.csv', index=False)
    y_train.to_csv(f'{config.DATA_PROCESSED_PATH}y_train.csv', index=False)
    y_test.to_csv(f'{config.DATA_PROCESSED_PATH}y_test.csv', index=False)

    print(f" Datos guardados en: {config.DATA_PROCESSED_PATH}")


def load_and_preprocess_data(file_path, target_column='demanda'):
    print("="*60)
    print(" INICIANDO PREPROCESAMIENTO DE DATOS")
    print("="*60)
    
    df = load_raw_data(file_path)
    
    df = clean_data(df)
    
    df = prepare_features(df)
    
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    
    save_processed_data(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*60)
    print(" PREPROCESAMIENTO COMPLETADO")
    print("="*60)
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    file_path = 'data/raw/ventas.csv'
    target_column = 'demanda'
    
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            file_path=file_path,
            target_column=target_column
        )
        print("\n RESUMEN FINAL:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
    except Exception as e:
        print(f"\n Error: {e}")