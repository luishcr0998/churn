# Telco Churn Prediction - Código Fonte

Este diretório contém os scripts Python que implementam o sistema de predição de churn para a Telco.

## Estrutura de Arquivos

- `__init__.py` - Define o pacote Python
- `data_processing.py` - Funções para processamento e transformação de dados
- `prediction.py` - Funções para fazer predições com o modelo treinado
- `main.py` - Script principal para executar o pipeline completo
- `example_usage.py` - Exemplos de como usar a API do sistema

## Uso Básico

### Executar o Pipeline Completo

Para executar o pipeline completo (pré-processamento, treinamento e predição):

```bash
python src/main.py --mode all
```

### Executar Apenas o Pré-processamento

```bash
python src/main.py --mode preprocess
```

### Executar Apenas Predições

Para fazer predições com um arquivo de entrada específico:

```bash
python src/main.py --mode predict --input-file caminho/para/dados.csv --output-dir caminho/para/resultados
```

### Parâmetros Disponíveis

- `--mode`: Modo de operação (`preprocess`, `train`, `predict`, `all`)
- `--input-file`: Caminho para o arquivo CSV de entrada
- `--output-dir`: Diretório para salvar os resultados
- `--threshold`: Limiar de probabilidade para classificar como churn (padrão: 0.5)

## Uso Avançado

### Utilização como API

Os módulos podem ser importados e utilizados como uma API em outros scripts Python:

```python
# Importar funções necessárias
from data_processing import clean_data, engineer_features, preprocess_new_data
from prediction import predict_churn, predict_single_customer, generate_prediction_report

# Exemplo: Predição para um único cliente
customer_data = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    # ... outros atributos do cliente
}
prediction = predict_single_customer(customer_data)
print(f"Probabilidade de churn: {prediction['churn_probability']:.4f}")

# Exemplo: Predição em lote
import pandas as pd
df = pd.read_csv('novos_clientes.csv')
results = predict_churn(df)
```

Para ver exemplos detalhados, consulte o arquivo `example_usage.py`.

### Configuração do Modelo

O sistema utiliza o modelo treinado salvo em `../models/final_model.pkl`. Para utilizar um modelo diferente, substitua este arquivo pelo seu modelo treinado.

### Customização do Pipeline

Para modificar o comportamento do pipeline de processamento de dados:

1. Abra o arquivo `data_processing.py`
2. Modifique as funções relevantes (por exemplo, `engineer_features()` para adicionar ou remover features)
3. Execute o pipeline novamente

## Dependências

O sistema depende das seguintes bibliotecas Python:

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn

As versões específicas estão listadas no arquivo `requirements.txt` na raiz do projeto.

## Solução de Problemas

### Erros Comuns

1. **FileNotFoundError**: Verifique se os diretórios `data`, `models` e `reports` existem na raiz do projeto.
2. **ModuleNotFoundError**: Verifique se instalou todas as dependências listadas em `requirements.txt`.
3. **Erro ao carregar o modelo**: Verifique se o modelo foi treinado e salvo corretamente.

### Logs e Depuração

Para obter mais informações durante a execução, você pode adicionar a variável de ambiente `DEBUG=1`:

```bash
DEBUG=1 python src/main.py --mode all
``` 