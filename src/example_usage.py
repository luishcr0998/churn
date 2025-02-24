"""
Exemplos de uso do sistema de predição de churn da Telco.

Este script contém exemplos práticos de como usar o sistema para
fazer predições de churn tanto para um único cliente quanto para
um conjunto de dados completo.
"""

import os
import pandas as pd
from prediction import predict_churn, predict_single_customer, batch_predict, generate_prediction_report

# Definir caminho para os dados de exemplo
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'telco_churn_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'reports', 'example_results')


def example_single_customer():
    """
    Exemplo de predição para um único cliente.
    """
    print("\n" + "="*50)
    print("Exemplo 1: Predição para um único cliente")
    print("="*50)
    
    # Dados de um novo cliente
    customer_data = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 36,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'Yes',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 95.70,
        'TotalCharges': 3455.20
    }
    
    # Fazer predição
    print("Dados do cliente:")
    for key, value in customer_data.items():
        print(f"  {key}: {value}")
        
    print("\nRealizando predição...")
    prediction = predict_single_customer(customer_data)
    
    print("\nResultado da predição:")
    print(f"  Probabilidade de churn: {prediction['churn_probability']:.4f}")
    print(f"  Previsão de churn: {'Sim' if prediction['predicted_churn'] else 'Não'}")
    print(f"  Segmento de risco: {prediction['risk_segment']}")
    
    # Recomendações com base no resultado
    print("\nRecomendação:")
    if prediction['risk_segment'] == 'Alto Risco':
        print("  Este cliente tem alto risco de churn. Recomenda-se ação imediata:")
        print("  - Oferecer desconto significativo por renovação de contrato")
        print("  - Propor upgrade para serviços premium com benefícios especiais")
        print("  - Agendar contato do gerente de contas para entender insatisfações")
    elif prediction['risk_segment'] == 'Médio Risco':
        print("  Este cliente tem risco moderado de churn. Recomenda-se:")
        print("  - Oferecer pequeno desconto ou promoção personalizada")
        print("  - Sugerir serviços adicionais que complementem seu perfil de uso")
        print("  - Enviar pesquisa de satisfação e follow-up")
    else:
        print("  Este cliente tem baixo risco de churn. Recomenda-se:")
        print("  - Manter serviço de qualidade")
        print("  - Explorar oportunidades de up-selling ou cross-selling")
        print("  - Incluir em programas de fidelidade e reconhecimento")


def example_batch_prediction():
    """
    Exemplo de predição em lote para um conjunto de dados.
    """
    print("\n" + "="*50)
    print("Exemplo 2: Predição em lote para um conjunto de dados")
    print("="*50)
    
    # Criar diretório de saída
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'batch_predictions.csv')
    
    print(f"Carregando dados de {DATA_PATH}...")
    
    # Carregar apenas as primeiras 100 linhas para exemplo
    df = pd.read_csv(DATA_PATH, nrows=100)
    print(f"Carregadas {len(df)} linhas de dados.")
    
    print("\nRealizando predições em lote...")
    results = batch_predict(df, output_path)
    
    print(f"\nPredições concluídas. Resultados salvos em {output_path}")
    print(f"Total de clientes analisados: {len(results)}")
    print(f"Clientes com previsão de churn: {results['predicted_churn'].sum()} ({results['predicted_churn'].mean()*100:.2f}%)")
    
    # Gerar relatório
    print("\nGerando relatório com visualizações...")
    generate_prediction_report(results, OUTPUT_DIR)
    
    print(f"\nRelatório gerado em {OUTPUT_DIR}")
    print(f"Abra {os.path.join(OUTPUT_DIR, 'prediction_report.html')} para visualizar o relatório completo.")


def example_api_workflow():
    """
    Exemplo de fluxo de trabalho completo da API.
    """
    print("\n" + "="*50)
    print("Exemplo 3: Fluxo de trabalho completo da API")
    print("="*50)
    
    # 1. Preparar dados de teste
    print("1. Preparando dados de teste...")
    # Carregar as primeiras 10 linhas para simular novos clientes
    new_customers = pd.read_csv(DATA_PATH, nrows=10)
    # Remover a coluna de churn para simular dados sem rótulo
    new_customers_unlabeled = new_customers.drop('Churn', axis=1)
    
    # 2. Fazer predições
    print("2. Realizando predições para novos clientes...")
    predictions = predict_churn(new_customers_unlabeled)
    
    # 3. Exibir resultados
    print("\n3. Resultados das predições:")
    print(predictions[['customerID', 'churn_probability', 'predicted_churn_label', 'risk_segment']].head())
    
    # 4. Agrupar resultados por segmento de risco
    risk_counts = predictions['risk_segment'].value_counts()
    print("\n4. Distribuição por segmento de risco:")
    for segment, count in risk_counts.items():
        print(f"  {segment}: {count} clientes ({count/len(predictions)*100:.1f}%)")
    
    # 5. Identificar clientes de alto risco
    high_risk = predictions[predictions['risk_segment'] == 'Alto Risco']
    print(f"\n5. Clientes de alto risco ({len(high_risk)}):")
    if not high_risk.empty:
        for _, row in high_risk.iterrows():
            print(f"  Cliente {row['customerID']}: {row['churn_probability']:.4f} probabilidade de churn")
    else:
        print("  Nenhum cliente de alto risco identificado.")


if __name__ == "__main__":
    print("="*50)
    print("Exemplos de uso do sistema de predição de churn")
    print("="*50)
    
    # Executar exemplos
    example_single_customer()
    example_batch_prediction()
    example_api_workflow()
    
    print("\n" + "="*50)
    print("Exemplos concluídos com sucesso!")
    print("="*50) 