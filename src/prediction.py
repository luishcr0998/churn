"""
Script para fazer predições de churn usando o modelo treinado.

Este script contém funções para carregar o modelo e fazer predições de churn
para novos clientes ou conjuntos de dados.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, Union, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Importar funções do módulo de processamento de dados
from data_processing import preprocess_new_data, load_data, clean_data, engineer_features

# Definir constantes
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'final_model.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')


def load_model() -> Tuple[Any, Dict]:
    """
    Carrega o modelo treinado e seus metadados.
    
    Returns:
        Tupla contendo o modelo e seus metadados.
    """
    # Verificar se o modelo existe
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")
    
    # Carregar o modelo
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # Carregar metadados do modelo
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    print(f"Modelo carregado: {metadata['model_name']}")
    print(f"Desempenho (ROC AUC): {metadata['roc_auc']:.4f}")
    
    return model, metadata


def predict_churn(data: Union[pd.DataFrame, str], 
                 threshold: float = 0.5) -> pd.DataFrame:
    """
    Faz predições de churn para um conjunto de dados.
    
    Args:
        data: DataFrame ou caminho para um arquivo CSV com dados de clientes.
        threshold: Limiar de probabilidade para classificar como churn (padrão: 0.5).
        
    Returns:
        DataFrame com as predições.
    """
    # Carregar dados, se necessário
    if isinstance(data, str):
        df = load_data(data)
    else:
        df = data.copy()
    
    # Verificar se há customerID no DataFrame
    has_customer_id = 'customerID' in df.columns
    customer_ids = df['customerID'].values if has_customer_id else None
    
    # Verificar se há coluna Churn no DataFrame
    has_churn = 'Churn' in df.columns
    actual_churn = None
    if has_churn:
        actual_churn = df['Churn'].map({'No': 0, 'Yes': 1}) if df['Churn'].dtype == 'object' else df['Churn']
    
    # Pré-processar os dados
    X_processed = preprocess_new_data(df)
    
    # Carregar o modelo
    model, _ = load_model()
    
    # Fazer predições
    y_prob = model.predict_proba(X_processed)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    # Criar DataFrame de resultados
    results = pd.DataFrame()
    
    if has_customer_id:
        results['customerID'] = customer_ids
    
    results['churn_probability'] = y_prob
    results['predicted_churn'] = y_pred
    results['predicted_churn_label'] = results['predicted_churn'].map({0: 'No', 1: 'Yes'})
    
    if has_churn:
        results['actual_churn'] = actual_churn
        results['correct_prediction'] = results['predicted_churn'] == results['actual_churn']
    
    # Classificar clientes de acordo com o risco de churn
    results['risk_segment'] = pd.cut(
        results['churn_probability'], 
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Baixo Risco', 'Médio Risco', 'Alto Risco']
    )
    
    return results


def batch_predict(input_path: str, output_path: str = None, 
                 threshold: float = 0.5) -> pd.DataFrame:
    """
    Realiza predições em lote para um arquivo de dados e salva os resultados.
    
    Args:
        input_path: Caminho para o arquivo CSV com dados de clientes.
        output_path: Caminho para salvar o arquivo de resultados (opcional).
        threshold: Limiar de probabilidade para classificar como churn (padrão: 0.5).
        
    Returns:
        DataFrame com as predições.
    """
    # Fazer predições
    results = predict_churn(input_path, threshold)
    
    # Salvar resultados, se solicitado
    if output_path:
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Salvar resultados
        results.to_csv(output_path, index=False)
        print(f"Resultados salvos em {output_path}")
    
    return results


def generate_prediction_report(results: pd.DataFrame, 
                             output_dir: str = None) -> None:
    """
    Gera um relatório de predições com visualizações.
    
    Args:
        results: DataFrame com os resultados das predições.
        output_dir: Diretório para salvar o relatório (opcional).
    """
    # Definir diretório de saída
    if output_dir is None:
        output_dir = os.path.join(REPORTS_DIR, 'predictions')
    
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Distribuição das probabilidades de churn
    plt.figure(figsize=(10, 6))
    sns.histplot(results['churn_probability'], bins=20, kde=True)
    plt.title('Distribuição das Probabilidades de Churn', fontsize=14)
    plt.xlabel('Probabilidade de Churn')
    plt.ylabel('Frequência')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'churn_probability_distribution.png'))
    plt.close()
    
    # 2. Contagem por segmento de risco
    plt.figure(figsize=(10, 6))
    risk_counts = results['risk_segment'].value_counts().sort_index()
    ax = sns.barplot(x=risk_counts.index, y=risk_counts.values)
    
    # Adicionar rótulos
    for i, count in enumerate(risk_counts):
        ax.text(i, count + 5, f'{count}', ha='center', fontsize=12)
    
    plt.title('Distribuição por Segmento de Risco de Churn', fontsize=14)
    plt.xlabel('Segmento de Risco')
    plt.ylabel('Contagem de Clientes')
    plt.ylim(0, max(risk_counts) * 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'risk_segment_distribution.png'))
    plt.close()
    
    # 3. Se tivermos dados reais, gerar matriz de confusão
    if 'actual_churn' in results.columns:
        plt.figure(figsize=(8, 6))
        conf_matrix = pd.crosstab(results['actual_churn'], results['predicted_churn'], 
                                  rownames=['Real'], colnames=['Predito'])
        
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Não Churn', 'Churn'],
                   yticklabels=['Não Churn', 'Churn'])
        plt.title('Matriz de Confusão', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Calcular métricas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(results['actual_churn'], results['predicted_churn'])
        precision = precision_score(results['actual_churn'], results['predicted_churn'])
        recall = recall_score(results['actual_churn'], results['predicted_churn'])
        f1 = f1_score(results['actual_churn'], results['predicted_churn'])
        roc_auc = roc_auc_score(results['actual_churn'], results['churn_probability'])
        
        # Salvar métricas
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        with open(os.path.join(output_dir, 'prediction_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
    
    # 4. Gerar relatório em HTML
    report_html = f"""
    <html>
    <head>
        <title>Relatório de Predição de Churn</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .risk-high {{ background-color: #ffcccc; }}
            .risk-medium {{ background-color: #ffffcc; }}
            .risk-low {{ background-color: #ccffcc; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Relatório de Predição de Churn</h1>
        
        <h2>Resumo</h2>
        <p>Total de clientes analisados: {len(results)}</p>
        <p>Clientes com previsão de churn: {results['predicted_churn'].sum()} ({results['predicted_churn'].mean()*100:.2f}%)</p>
        
        <h2>Distribuição por Segmento de Risco</h2>
        <table>
            <tr>
                <th>Segmento de Risco</th>
                <th>Número de Clientes</th>
                <th>Porcentagem</th>
            </tr>
    """
    
    # Adicionar linhas para cada segmento de risco
    for segment, count in risk_counts.items():
        pct = count / len(results) * 100
        report_html += f"""
            <tr>
                <td>{segment}</td>
                <td>{count}</td>
                <td>{pct:.2f}%</td>
            </tr>
        """
    
    report_html += """
        </table>
        
        <h2>Visualizações</h2>
        <h3>Distribuição das Probabilidades de Churn</h3>
        <img src="churn_probability_distribution.png" alt="Distribuição das Probabilidades de Churn">
        
        <h3>Distribuição por Segmento de Risco</h3>
        <img src="risk_segment_distribution.png" alt="Distribuição por Segmento de Risco">
    """
    
    # Se tivermos dados reais, adicionar matriz de confusão e métricas
    if 'actual_churn' in results.columns:
        report_html += f"""
        <h3>Matriz de Confusão</h3>
        <img src="confusion_matrix.png" alt="Matriz de Confusão">
        
        <h2>Métricas de Desempenho</h2>
        <table>
            <tr>
                <th>Métrica</th>
                <th>Valor</th>
            </tr>
            <tr>
                <td>Acurácia</td>
                <td>{accuracy:.4f}</td>
            </tr>
            <tr>
                <td>Precisão</td>
                <td>{precision:.4f}</td>
            </tr>
            <tr>
                <td>Recall</td>
                <td>{recall:.4f}</td>
            </tr>
            <tr>
                <td>F1-Score</td>
                <td>{f1:.4f}</td>
            </tr>
            <tr>
                <td>ROC AUC</td>
                <td>{roc_auc:.4f}</td>
            </tr>
        </table>
        """
    
    # Top clientes com maior risco de churn
    top_risk_clients = results.sort_values('churn_probability', ascending=False).head(20)
    
    report_html += """
        <h2>Top 20 Clientes com Maior Risco de Churn</h2>
        <table>
            <tr>
                <th>ID do Cliente</th>
                <th>Probabilidade de Churn</th>
                <th>Previsão</th>
                <th>Segmento de Risco</th>
            </tr>
    """
    
    for _, row in top_risk_clients.iterrows():
        risk_class = ""
        if row['risk_segment'] == 'Alto Risco':
            risk_class = "risk-high"
        elif row['risk_segment'] == 'Médio Risco':
            risk_class = "risk-medium"
        else:
            risk_class = "risk-low"
        
        customer_id = row['customerID'] if 'customerID' in row else f"Cliente #{_}"
        
        report_html += f"""
            <tr class="{risk_class}">
                <td>{customer_id}</td>
                <td>{row['churn_probability']:.4f}</td>
                <td>{row['predicted_churn_label']}</td>
                <td>{row['risk_segment']}</td>
            </tr>
        """
    
    report_html += """
        </table>
        
        <h2>Recomendações</h2>
        <p>Com base nas predições, sugerimos as seguintes ações:</p>
        <ol>
            <li>Para clientes de alto risco (probabilidade > 60%): Implementar programa de retenção imediato, oferecendo benefícios personalizados.</li>
            <li>Para clientes de médio risco (probabilidade entre 30% e 60%): Monitorar de perto e oferecer melhorias nos serviços atuais.</li>
            <li>Para clientes de baixo risco (probabilidade < 30%): Manter a qualidade do serviço e identificar oportunidades de up-selling.</li>
        </ol>
        
        <p><i>Relatório gerado automaticamente pelo sistema de predição de churn Telco.</i></p>
    </body>
    </html>
    """
    
    # Salvar relatório HTML
    with open(os.path.join(output_dir, 'prediction_report.html'), 'w') as f:
        f.write(report_html)
    
    print(f"Relatório de predição gerado em {output_dir}")


def predict_single_customer(customer_data: Dict) -> Dict:
    """
    Faz predição de churn para um único cliente.
    
    Args:
        customer_data: Dicionário com dados do cliente.
        
    Returns:
        Dicionário com a predição.
    """
    # Converter o dicionário para DataFrame
    df = pd.DataFrame([customer_data])
    
    # Fazer predição
    results = predict_churn(df)
    
    # Extrair resultado
    prediction = {
        'churn_probability': float(results['churn_probability'].iloc[0]),
        'predicted_churn': bool(results['predicted_churn'].iloc[0]),
        'risk_segment': results['risk_segment'].iloc[0]
    }
    
    return prediction


def run_prediction_pipeline(input_path: str = None, output_dir: str = None, threshold: float = 0.5) -> None:
    """
    Executa o pipeline completo de predição.
    
    Args:
        input_path: Caminho para o arquivo CSV com dados de clientes. Se None, usa o arquivo padrão.
        output_dir: Diretório para salvar os resultados. Se None, usa o diretório padrão.
        threshold: Limiar de probabilidade para classificar como churn (padrão: 0.5).
    """
    print("Iniciando pipeline de predição...")
    
    # Definir caminho de entrada padrão, se não fornecido
    if input_path is None:
        input_path = os.path.join(DATA_DIR, 'telco_churn_data.csv')
    
    # Definir diretório de saída padrão, se não fornecido
    if output_dir is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(REPORTS_DIR, f'predictions_{timestamp}')
    
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Definir caminho de saída
    output_path = os.path.join(output_dir, 'churn_predictions.csv')
    
    # 1. Fazer predições em lote
    results = batch_predict(input_path, output_path, threshold)
    
    # 2. Gerar relatório
    generate_prediction_report(results, output_dir)
    
    print("\nPipeline de predição concluído com sucesso!")
    print(f"Resultados salvos em {output_path}")
    print(f"Relatório gerado em {output_dir}/prediction_report.html")


if __name__ == "__main__":
    # Executar o pipeline completo de predição
    run_prediction_pipeline() 