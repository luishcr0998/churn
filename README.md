# Projeto de Predição de Churn em Telecomunicações

## Descrição do Projeto
Este projeto tem como objetivo desenvolver um modelo de machine learning para prever a probabilidade de churn (cancelamento) de clientes de uma empresa de telecomunicações. A predição de churn é fundamental para empresas de telecomunicações, pois permite identificar clientes com risco de cancelamento e implementar estratégias de retenção proativas.

## Estrutura do Projeto
```
telco_churn_project/
│
├── data/ - Armazena os dados brutos e processados
│
├── notebooks/ - Jupyter notebooks para análise exploratória e modelagem
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_model_evaluation.ipynb
│
├── models/ - Modelos treinados e serializados
│
├── reports/ - Relatórios, visualizações e resultados do projeto
│   ├── figures/ - Visualizações geradas durante a análise
│   └── metrics/ - Métricas de desempenho dos modelos
│
├── src/ - Código fonte do projeto
│   ├── data_processing.py - Scripts para processamento de dados
│   ├── feature_engineering.py - Scripts para engenharia de features
│   ├── model_training.py - Scripts para treinamento de modelos
│   └── evaluation.py - Scripts para avaliação de modelos
│
├── requirements.txt - Lista de dependências do projeto
└── README.md - Documentação do projeto
```

## Conjunto de Dados
O conjunto de dados utilizado contém informações de clientes de telecomunicações com as seguintes características:
- Dados demográficos (gênero, status civil, presença de dependentes)
- Histórico do cliente (tempo de contrato, serviços contratados)
- Informações de cobrança (forma de pagamento, valor mensal, valor total)
- A variável alvo "Churn" indica se o cliente cancelou o serviço (Sim/Não)

## Metodologia
1. **Análise Exploratória de Dados (EDA)**
   - Compreensão da distribuição das variáveis
   - Identificação de correlações
   - Tratamento de valores ausentes ou anômalos
   - Visualização de padrões e insights

2. **Pré-processamento de Dados**
   - Codificação de variáveis categóricas
   - Normalização/padronização de variáveis numéricas
   - Tratamento do desbalanceamento de classes
   - Engenharia de features

3. **Modelagem**
   - Divisão dos dados em conjuntos de treinamento e teste
   - Implementação e comparação de diferentes algoritmos:
     - Random Forest
     - XGBoost
     - LightGBM
     - Redes Neurais
   - Otimização de hiperparâmetros
   - Ensemble de modelos

4. **Avaliação**
   - Métricas de desempenho (precisão, recall, F1-score, AUC-ROC)
   - Validação cruzada
   - Matriz de confusão
   - Análise de importância de features

5. **Implantação**
   - Serialização do modelo final
   - Documentação do pipeline de predição
   - Recomendações para integração com sistemas de CRM

## Instalação e Uso

1. Clone o repositório
2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```
3. Execute os notebooks na ordem especificada

## Referências
- Melhores práticas para predição de churn em telecomunicações
- Técnicas de modelagem para dados desbalanceados
- Estratégias de retenção de clientes baseadas em ML 