"""
Script principal para executar o pipeline completo do projeto de predição de churn.

Este script coordena todas as etapas do projeto, desde o pré-processamento dos dados até a geração
de predições e relatórios.
"""

import os
import argparse
import time
from typing import Dict, Any
import pandas as pd

# Importar funções dos demais módulos
from data_processing import run_preprocessing_pipeline, load_data, clean_data, engineer_features
from prediction import run_prediction_pipeline, predict_single_customer


def parse_arguments() -> Dict[str, Any]:
    """
    Analisa os argumentos da linha de comando.
    
    Returns:
        Dicionário com os argumentos.
    """
    parser = argparse.ArgumentParser(description='Telco Churn Prediction Pipeline')
    
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'predict', 'all'], 
                      default='all', help='Modo de operação')
    
    parser.add_argument('--input-file', type=str,
                      help='Caminho para o arquivo CSV de entrada')
    
    parser.add_argument('--output-dir', type=str,
                      help='Diretório para salvar os resultados')
    
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Limiar de probabilidade para classificar como churn')
    
    args = parser.parse_args()
    
    return vars(args)


def create_output_directory(output_dir: str = None) -> str:
    """
    Cria um diretório de saída com timestamp.
    
    Args:
        output_dir: Diretório base para saída. Se None, usa o diretório padrão.
        
    Returns:
        Caminho completo para o diretório criado.
    """
    if output_dir is None:
        # Diretório padrão para resultados
        base_dir = os.path.dirname(os.path.dirname(__file__))
        results_dir = os.path.join(base_dir, 'reports')
    else:
        results_dir = output_dir
    
    # Criar diretório com timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(results_dir, f'results_{timestamp}')
    
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def run_train_pipeline() -> None:
    """
    Executa o pipeline de treinamento.
    
    Nota: A implementação real do treinamento está nos notebooks. Esta função
    apenas fornece uma referência para integração futura.
    """
    print("O pipeline de treinamento é executado nos notebooks.")
    print("Execute os notebooks na seguinte ordem:")
    print("1. 01_exploratory_data_analysis.ipynb")
    print("2. 02_feature_engineering.ipynb")
    print("3. 03_model_development.ipynb")
    print("4. 04_model_evaluation.ipynb")
    
    # Aqui você poderia importar e executar funções de um módulo de treinamento
    # Por exemplo: from model_training import train_model
    # train_model()


def run_full_pipeline(args: Dict[str, Any]) -> None:
    """
    Executa o pipeline completo do projeto.
    
    Args:
        args: Argumentos da linha de comando.
    """
    start_time = time.time()
    print("="*80)
    print("Iniciando pipeline completo de predição de churn da Telco")
    print("="*80)
    
    # Definir diretório de saída
    output_dir = args.get('output_dir')
    if output_dir is None:
        output_dir = create_output_directory()
    
    # 1. Pré-processamento
    print("\n" + "="*30 + " Etapa 1: Pré-processamento " + "="*30)
    run_preprocessing_pipeline()
    
    # 2. Treinamento (simulado)
    print("\n" + "="*30 + " Etapa 2: Treinamento " + "="*30)
    run_train_pipeline()
    
    # 3. Predição
    print("\n" + "="*30 + " Etapa 3: Predição " + "="*30)
    input_file = args.get('input_file')
    threshold = args.get('threshold', 0.5)
    run_prediction_pipeline(input_file, output_dir, threshold)
    
    elapsed_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"Pipeline completo concluído em {elapsed_time:.2f} segundos")
    print(f"Resultados salvos em: {output_dir}")
    print("="*80)


def main() -> None:
    """
    Função principal que coordena a execução do pipeline.
    """
    # Analisar argumentos
    args = parse_arguments()
    
    # Executar o modo selecionado
    if args['mode'] == 'preprocess':
        run_preprocessing_pipeline()
    elif args['mode'] == 'train':
        run_train_pipeline()
    elif args['mode'] == 'predict':
        output_dir = args.get('output_dir')
        if output_dir is None:
            output_dir = create_output_directory()
        run_prediction_pipeline(args.get('input_file'), output_dir, args.get('threshold', 0.5))
    elif args['mode'] == 'all':
        run_full_pipeline(args)


if __name__ == "__main__":
    main() 