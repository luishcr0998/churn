"""
Script para processamento e transformação de dados para o modelo de predição de churn.

Este script contém funções para carregar, limpar, transformar e preparar os dados 
para o modelo de predição de churn da Telco.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Union, Dict, Any
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Definir constantes
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'telco_churn_data.csv')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
PREPROCESSOR_PATH = os.path.join(PROCESSED_DATA_DIR, 'preprocessor.pkl')


def load_data(file_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Carrega os dados do arquivo CSV.
    
    Args:
        file_path: Caminho para o arquivo CSV.
        
    Returns:
        DataFrame com os dados carregados.
    """
    df = pd.read_csv(file_path)
    print(f"Dados carregados com {df.shape[0]} linhas e {df.shape[1]} colunas.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a limpeza dos dados.
    
    Args:
        df: DataFrame com os dados a serem limpos.
        
    Returns:
        DataFrame com os dados limpos.
    """
    # Criar uma cópia para não modificar o original
    df_clean = df.copy()
    
    # Verificar valores ausentes
    missing_values = df_clean.isnull().sum()
    print(f"Valores ausentes antes da limpeza:\n{missing_values[missing_values > 0]}")
    
    # Converter TotalCharges para numérico, se necessário
    if df_clean['TotalCharges'].dtype == 'object':
        # Substituir espaços vazios por NaN
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # Preencher valores NaN em TotalCharges
        mask_new_customers = df_clean['tenure'] == 0
        
        # Para novos clientes, TotalCharges deve ser igual a MonthlyCharges
        df_clean.loc[mask_new_customers & df_clean['TotalCharges'].isna(), 'TotalCharges'] = \
            df_clean.loc[mask_new_customers & df_clean['TotalCharges'].isna(), 'MonthlyCharges']
        
        # Para os demais casos (se houver), usar a média
        df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].mean(), inplace=True)
    
    # Converter SeniorCitizen para tipo categórico
    df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype('object')
    df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    # Verificar valores ausentes após a limpeza
    missing_values_after = df_clean.isnull().sum()
    print(f"Valores ausentes após a limpeza:\n{missing_values_after[missing_values_after > 0]}")
    
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a engenharia de features.
    
    Args:
        df: DataFrame com os dados limpos.
        
    Returns:
        DataFrame com as novas features.
    """
    # Criar uma cópia para não modificar o original
    df_features = df.copy()
    
    # Remover a coluna de ID, pois não é relevante para o modelo
    if 'customerID' in df_features.columns:
        df_features = df_features.drop('customerID', axis=1)
    
    # Criar feature para agrupar o tempo de contrato
    df_features['tenure_group'] = pd.cut(df_features['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                                        labels=['0-12 meses', '13-24 meses', '25-36 meses', 
                                                '37-48 meses', '49-60 meses', '61-72 meses'])
    
    # Criar feature para contagem de serviços contratados
    services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Inicializar contagem de serviços
    df_features['num_services'] = 0
    
    # Contar serviços ativos
    for service in services:
        df_features['num_services'] += (df_features[service] == 'Yes').astype(int)
    
    # Criar feature para serviços de internet
    internet_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Inicializar contagem de serviços de internet
    df_features['num_internet_services'] = 0
    
    # Contar serviços de internet ativos
    for service in internet_services:
        df_features['num_internet_services'] += (df_features[service] == 'Yes').astype(int)
    
    # Criar feature para taxa de valor mensal por serviço
    df_features['avg_cost_per_service'] = df_features['MonthlyCharges'] / (df_features['num_services'] + 1)
    
    # Criar feature para categorias de valor mensal
    df_features['monthly_charge_category'] = pd.qcut(df_features['MonthlyCharges'], q=4, 
                                                    labels=['Baixo', 'Médio-Baixo', 'Médio-Alto', 'Alto'])
    
    # Criar feature que indica se o cliente possui serviços de streaming
    df_features['has_streaming'] = ((df_features['StreamingTV'] == 'Yes') | 
                                    (df_features['StreamingMovies'] == 'Yes')).astype(int)
    
    # Criar feature que indica se o cliente possui serviços de segurança
    df_features['has_security'] = ((df_features['OnlineSecurity'] == 'Yes') | 
                                    (df_features['OnlineBackup'] == 'Yes') | 
                                    (df_features['DeviceProtection'] == 'Yes') | 
                                    (df_features['TechSupport'] == 'Yes')).astype(int)
    
    # Criar feature que combina tipo de contrato e método de pagamento
    df_features['contract_payment'] = df_features['Contract'] + '_' + df_features['PaymentMethod'].apply(lambda x: x.replace(' ', '_'))
    
    print(f"Engenharia de features concluída. Número de features: {df_features.shape[1]}")
    return df_features


def create_preprocessor(df: pd.DataFrame, target_col: str = 'Churn') -> ColumnTransformer:
    """
    Cria o pipeline de pré-processamento para os dados.
    
    Args:
        df: DataFrame com as features.
        target_col: Nome da coluna alvo.
        
    Returns:
        ColumnTransformer com o pipeline de pré-processamento.
    """
    # Separar X e y
    X = df.drop(target_col, axis=1)
    
    # Identificar variáveis categóricas e numéricas
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Variáveis categóricas: {len(categorical_cols)}")
    print(f"Variáveis numéricas: {len(numerical_cols)}")
    
    # 1. Para variáveis numéricas: imputação e padronização
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 2. Para variáveis categóricas: one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combinando transformadores em um preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Salvar informações das features
    feature_info = {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols
    }
    
    # Criar diretório, se não existir
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Salvar informações das features
    with open(os.path.join(PROCESSED_DATA_DIR, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_info, f)
    
    return preprocessor


def prepare_data_for_training(df: pd.DataFrame, target_col: str = 'Churn', 
                              test_size: float = 0.2, random_state: int = 42, 
                              apply_smote: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepara os dados para treinamento, incluindo divisão em treino/teste e pré-processamento.
    
    Args:
        df: DataFrame com os dados.
        target_col: Nome da coluna alvo.
        test_size: Proporção do conjunto de teste.
        random_state: Seed para reprodutibilidade.
        apply_smote: Se True, aplica SMOTE no conjunto de treinamento.
        
    Returns:
        X_train, X_test, y_train, y_test: Conjuntos de treinamento e teste.
    """
    # Separar X e y
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Converter a variável alvo para binária
    y = y.map({'No': 0, 'Yes': 1})
    
    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, stratify=y)
    
    print(f"Dimensões do conjunto de treinamento (X_train): {X_train.shape}")
    print(f"Dimensões do conjunto de teste (X_test): {X_test.shape}")
    
    # Criar e ajustar o preprocessor
    preprocessor = create_preprocessor(df, target_col)
    
    # Aplicar o pré-processamento
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Dimensões de X_train após pré-processamento: {X_train_processed.shape}")
    print(f"Dimensões de X_test após pré-processamento: {X_test_processed.shape}")
    
    # Salvar o preprocessor
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    with open(PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Aplicar SMOTE para tratar o desbalanceamento, se solicitado
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
        
        print(f"Dimensões de X_train após SMOTE: {X_train_resampled.shape}")
        
        # Salvar dados processados
        np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train_processed.npy'), X_train_processed)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test_processed.npy'), X_test_processed)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
        
        # Salvar dados balanceados com SMOTE
        np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train_resampled.npy'), X_train_resampled)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train_resampled.npy'), y_train_resampled)
        
        return X_train_resampled, X_test_processed, y_train_resampled, y_test
    else:
        # Salvar dados processados
        np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train_processed.npy'), X_train_processed)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test_processed.npy'), X_test_processed)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
        
        return X_train_processed, X_test_processed, y_train, y_test


def preprocess_new_data(df: pd.DataFrame) -> np.ndarray:
    """
    Pré-processa novos dados usando o preprocessor salvo.
    
    Args:
        df: DataFrame com os novos dados.
        
    Returns:
        array com os dados pré-processados.
    """
    # Limpar e realizar engenharia de features
    df_clean = clean_data(df)
    df_features = engineer_features(df_clean)
    
    # Remover a coluna de Churn se existir
    if 'Churn' in df_features.columns:
        df_features = df_features.drop('Churn', axis=1)
    
    # Carregar o preprocessor
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Aplicar o pré-processamento
    X_processed = preprocessor.transform(df_features)
    
    return X_processed


def run_preprocessing_pipeline() -> None:
    """
    Executa o pipeline completo de pré-processamento.
    """
    print("Iniciando pipeline de pré-processamento...")
    
    # 1. Carregar dados
    df = load_data()
    
    # 2. Limpar dados
    df_clean = clean_data(df)
    
    # 3. Engenharia de features
    df_features = engineer_features(df_clean)
    
    # 4. Preparar dados para treinamento
    X_train, X_test, y_train, y_test = prepare_data_for_training(df_features)
    
    print("\nPipeline de pré-processamento concluído com sucesso!")
    print(f"Dados de treinamento: {X_train.shape}")
    print(f"Dados de teste: {X_test.shape}")


if __name__ == "__main__":
    # Executar o pipeline completo
    run_preprocessing_pipeline() 