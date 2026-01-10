# Tech Challenge Deep Learning - Previsão de Séries Temporais Financeiras com LSTM

**Tech Challenge 4 - Pós Graduação em Machine Learning Engineering**

Aplicação produtiva para previsão de séries temporais financeiras usando **LSTM (Long Short-Term Memory)** com hiperparâmetros otimizados via **Ray Tune**, experimentos rastreados em **MLflow** e API REST servida via **FastAPI**.

---

## 📋 Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Instalação e Setup](#instalação-e-setup)
- [Como Executar Localmente](#como-executar-localmente)
- [Scripts de Utilidade](#scripts-de-utilidade)
- [API REST](#api-rest)
- [Conceitos Técnicos](#conceitos-técnicos)
- [Modelo Campeão](#modelo-campeão)
- [Roadmap Futuro](#roadmap-futuro)
- [Contributing](#contributing)

---

## 🎯 Visão Geral

Este projeto implementa um **pipeline completo de ML** para previsão de preços de ações financeiras usando LSTMs. O fluxo típico é:

1. **Preparação de Dados**: Download automático de dados de ações via yfinance
2. **Processamento em Estratégias**: Múltiplas estratégias de processamento de dados (single/multiple tickers)
3. **Otimização de Hiperparâmetros**: Ray Tune executa combinações de parâmetros em paralelo
4. **Rastreamento de Experimentos**: MLflow registra todas as métricas, parâmetros e artefatos
5. **Seleção do Melhor Modelo**: Script automático identifica o modelo campeão
6. **Inferência**: API REST para fazer previsões com o modelo treinado

### Principais Características

- ✅ **Arquitetura Modular**: Padrão Strategy para flexibilidade de algoritmos
- ✅ **Rastreamento Completo**: MLflow para auditoria e reprodutibilidade
- ✅ **Otimização Automática**: Ray Tune para busca de hiperparâmetros distribuída
- ✅ **API RESTful**: FastAPI para servir previsões em produção
- ✅ **PyTorch Lightning**: Treinamento simplificado e reproducível com PyTorch
- ✅ **Monitoramento**: Endpoint para acompanhar latência e saúde da inferência

---

## 🏗️ Arquitetura do Projeto

```
tech-challenge-deep-learning/
│
├── src/                           # Código principal
│   ├── api/                       # FastAPI application
│   │   ├── routes/
│   │   │   ├── train.py          # Treinar modelo com config campeã
│   │   │   ├── infer.py          # Fazer previsões
│   │   │   ├── monitor.py        # Status e métricas
│   │   │   └── config.py         # Retornar config carregada
│   │   └── app.py                # Aplicação FastAPI principal
│   │
│   ├── data/                      # Processamento de dados
│   │   ├── data.py               # DataPipeline + DataStrategy (abstração)
│   │   └── scaler.py             # TimeSeriesScaler
│   │
│   ├── models/                    # Arquitetura neural
│   │   ├── lstm.py               # LSTM + LSTMFactory
│   │   └── lstm_params.py        # Parâmetros do modelo
│   │
│   ├── training/                  # Lógica de treinamento
│   │   ├── model.py              # LSTMLightningModule + TrainingStrategy
│   │   ├── metrics.py            # MAE, RMSE, MAPE
│   │   └── trainer.py            # TrainerContext (orquestrador)
│   │
│   ├── inference/                 # Predição em produção
│   │   └── predictor.py          # Função predict() principal
│   │
│   ├── services/                  # Camada de aplicação
│   │   └── monitoring_service.py # Rastrear latência/stats
│   │
│   └── utils/                     # Utilitários
│       └── model_loader.py       # Carregar config, modelo, scaler
│
├── scripts/                       # Scripts de execução
│   ├── download_data.py          # Baixa dados de ações (yfinance)
│   ├── run_ray_experiments.py    # Executa Ray Tune com múltiplas estratégias
│   ├── champion_selector.py      # Seleciona melhor run, salva best_config.yaml
│   ├── smoke_train.py            # Teste rápido de treinamento
│   └── smoke_test_data_pipeline.py # Teste rápido de pipeline de dados
│
├── configs/                       # Configurações YAML
│   ├── best_config.yaml          # ⭐ Config do modelo campeão (gerado)
│   └── ray_experiments.yaml      # Parâmetros para Ray Tune
│
├── artifacts/                     # Artefatos de treinamento
│   └── models/
│       └── model_final.pt        # ⭐ Peso do modelo treinado
│
├── mlruns/                        # MLflow tracking
│   └── [experiment_id]/          # Armazena experimentos e runs
│
├── tests/                         # Testes automatizados
│   ├── conftest.py
│   └── *.py
│
├── requirements.txt               # Dependências Python
├── README.md                      # Este arquivo
├── CONTRIBUTING.md                # Guia de contribuição
└── .gitignore
```

### Fluxo de Dados

```
yfinance (dados brutos)
    ↓
DataStrategy (processamento)
    ├→ NoProcessingSingle      (1 ticker, sem engenharia)
    ├→ NoProcessingMultiple    (N tickers, sem engenharia)
    ├→ RangeSingle            (1 ticker + features normalizadas)
    └→ RangeMultiple           (N tickers + features normalizadas)
    ↓
DataPipeline (batch creation)
    ↓
LSTMLightningModule + PyTorch Lightning Trainer
    ↓
MLflow (log metrics/params/artifacts)
    ↓
model_final.pt + scaler_*.pkl + best_config.yaml
    ↓
API /infer (predição em tempo real)
```

---

## 🔧 Tecnologias e Justificativas

### **PyTorch Lightning** 🔥

**Por que usamos?**

- **Simplicidade**: Reduz 50% do boilerplate de treinamento PyTorch puro
- **Reprodutibilidade**: Gerencia seeds, logging e checkpoints automaticamente
- **Escalabilidade**: Suporta multi-GPU/TPU com uma linha de configuração
- **Integração MLflow**: Logger nativo para rastreamento de experimentos

**Exemplo**: Sem Lightning, teríamos ~500 linhas de code para train/val loops. Com Lightning: ~100.

### **Ray Tune** 🎯

**Por que usamos?**

- **Busca Distribuída**: Executa múltiplas combinações de hiperparâmetros em paralelo
- **Escalabilidade**: Funciona em cluster com centenas de workers
- **Early Stopping**: Cancela trials ruins automaticamente (Hyperband)
- **Integração MLflow**: Registra cada trial como um run separado

**Exemplo**: 100 combinações levaria horas sequencialmente → minutos em paralelo.

### **MLflow** 📊

**Por que usamos?**

- **Rastreabilidade**: Cada experimento, run, métrica e artefato é registrado
- **Reprodutibilidade**: Recupera exatamente quais parâmetros geraram qual resultado
- **Seleção Automática**: `champion_selector.py` encontra o melhor run facilmente
- **UI Web**: Visualiza experimentos via `mlflow ui`

**Exemplo**: Sem MLflow, cada treinamento seria "uma caixa preta". Com MLflow, sabemos:
- Quais parâmetros usamos
- Quais métricas obtivemos
- Onde estão os artefatos (modelo, scaler)

---

## 📦 Instalação e Setup

### Requisitos

- **Python >= 3.13**
- `pip` e `venv` instalados

### Setup Local

```bash
# Clone o repositório
git clone <repo-url>
cd tech-challenge-deep-learning

# Crie ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instale dependências
pip install -r requirements.txt

# Valide a instalação (rodando testes)
pytest -q
```

### Dependências Principais

```
torch>=2.0.0             # Deep learning framework
pytorch-lightning>=2.0   # Treinamento simplificado
mlflow>=2.0              # Tracking de experimentos
ray[tune]>=2.0           # Otimização de hiperparâmetros
fastapi>=0.100           # API REST
uvicorn>=0.23            # ASGI server
pandas>=2.0              # Manipulação de dados
yfinance>=0.2.30         # Download de dados financeiros
scikit-learn>=1.3        # Utilities (scalers, metrics)
```

Ver [requirements.txt](requirements.txt) para lista completa.

---

## 🚀 Como Executar Localmente

### 1️⃣ Baixar Dados (Opcional)

Se não houver dados em `data/raw/AAPL.csv`:

```bash
python -m scripts.download_data
```

### 2️⃣ Executar Ray Tune para Otimizar Hiperparâmetros

Executa todas as combinações de estratégias e parâmetros em paralelo:

```bash
python -m scripts.run_ray_experiments
```

**Saída esperada:**
- Múltiplos runs registrados em MLflow
- Artefatos salvos em `mlruns/`
- Histórico de experimentos consultável via `mlflow ui`

### 3️⃣ Selecionar Modelo Campeão

Identifica o melhor run e salva configuração:

```bash
python -m scripts.champion_selector
```

**Saída esperada:**
- `configs/best_config.yaml` criado
- Melhor run ID exibido no console
- Métricas do campeão mostradas

### 4️⃣ Iniciar API REST

```bash
# Desenvolvimento com reload automático
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

**Saída esperada:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

Acesse a documentação interativa: **http://localhost:8000/docs**

### 5️⃣ (Opcional) Visualizar Experimentos MLflow

Em outro terminal:

```bash
mlflow server --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000
```

Acesse: **http://127.0.0.1:5000**

---

## 📝 Scripts de Utilidade

### `download_data.py`

**Responsabilidade**: Baixar dados históricos de ações

```bash
python -m scripts.download_data
```

**O que faz**:
- Download via yfinance para ticker especificado (padrão: AAPL)
- Salva CSV em `data/raw/{TICKER}.csv`
- Valida colunas obrigatórias (High, Low, Close, Volume)
- Implementa cache local (não faz download repetido)

**Exemplo de uso em código**:
```python
from scripts.download_data import download
download(ticker="AAPL", start="2020-01-01")
```

---

### `run_ray_experiments.py`

**Responsabilidade**: Executar combinações de hiperparâmetros com Ray Tune

```bash
python -m scripts.run_ray_experiments
```

**O que faz**:
1. Lê `configs/ray_experiments.yaml`
2. Cria produto cartesiano de todos os parâmetros
3. Para cada combinação:
   - Instancia uma estratégia de treinamento (NoProcessingSingle, etc.)
   - Cria DataPipeline correspondente
   - Executa treinamento via PyTorch Lightning
   - Loga métricas em MLflow
4. Retorna histórico de todos os runs

**Estratégias Testadas**:
- `NoProcessingSingleStrategy`: 1 ticker, sem engenharia de features
- `NoProcessingMultipleStrategy`: N tickers, sem engenharia
- `RangeSingleStrategy`: 1 ticker com normalização de features
- `RangeMultipleStrategy`: N tickers com normalização

**Exemplo de saída**:
```
Trial 1/100: NoProcessingSingleStrategy
  ├─ val_rmse: 2.34
  ├─ val_mae: 1.89
  └─ Run ID: abc123def456

Trial 2/100: NoProcessingMultipleStrategy
  ├─ val_rmse: 2.12 ✓ (melhor até agora)
  ├─ val_mae: 1.72
  └─ Run ID: xyz789uvw012

...
```

---

### `champion_selector.py`

**Responsabilidade**: Selecionar o melhor modelo e salvar sua configuração

```bash
python -m scripts.champion_selector
```

**O que faz**:
1. Query MLflow por todos os runs do experimento
2. Ordena por `val_rmse` (menor é melhor)
3. Extrai parâmetros do melhor run
4. Reconstrói configuração semântica
5. Salva em `configs/best_config.yaml`

**Estrutura do best_config.yaml**:
```yaml
metadata:
  selected_on: "2024-01-09 14:30:45"
  experiment_name: "lstm_strategy_experiments"
  run_id: "abc123..."
  metric: "val_rmse"
  metric_value: 2.12
  strategy: "NoProcessingMultipleStrategy"

model:
  type: "LSTM"
  input_size: 4
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
  output_size: 1
  layer_config: ["LSTM", "Linear"]

training:
  learning_rate: 0.001
  batch_size: 32
  num_epochs: 50
  shuffle: true

data:
  tickers: ["AAPL", "MSFT"]
  period: "1y"
  seq_len: 20
  train_ratio: 0.7
  val_ratio: 0.15
  scaler: "TimeSeriesScaler"
```

---

### `smoke_train.py` e `smoke_test_data_pipeline.py`

**Responsabilidade**: Testes rápidos para validação

**smoke_train.py**:
- Treina modelo com configuração mínima em ~30 segundos
- Valida pipeline completo de treinamento
- Não registra em MLflow

**smoke_test_data_pipeline.py**:
- Testa carregamento e processamento de dados
- Valida formato de tensores
- Detecta problemas em etapa inicial

**Uso**:
```bash
pytest tests/               # Testes completos
python -m scripts.smoke_train
python -m scripts.smoke_test_data_pipeline
```

---

## 🔌 API REST

A API é servida via **FastAPI** e documentada automaticamente via **Swagger**.

### Iniciar Servidor

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

#### **[GET] `/`**

Retorna informações gerais da API.

**Exemplo de Resposta**:
```json
{
  "title": "ML Training API",
  "description": "API for training LSTM models",
  "version": "1.0.0"
}
```

---

#### **[POST] `/train/`**

Treina modelo usando configuração do modelo campeão (`best_config.yaml`).

**Pré-requisito**: `champion_selector.py` deve ter sido executado.

**Payload**: Nenhum (usa configuração salva)

**Exemplo de Requisição**:
```bash
curl -X POST http://localhost:8000/train/
```

**Exemplo de Resposta** (após conclusão):
```json
{
  "status": "success",
  "message": "Training completed",
  "metrics": {
    "final_train_loss": 0.0234,
    "final_val_loss": 0.0456,
    "val_rmse": 2.34,
    "val_mae": 1.89
  },
  "model_path": "artifacts/models/model_final.pt",
  "training_duration_seconds": 245.67
}
```

**Possíveis Erros**:
- `404`: best_config.yaml não encontrado
- `400`: Parâmetros inválidos
- `500`: Erro durante treinamento

---

#### **[POST] `/infer/`**

Faz previsão com modelo treinado.

**Payload**:
```json
{
  "sequence": [[1.2, 2.3, 3.4, 4.5], [1.5, 2.6, 3.7, 4.8], ...]
}
```

Aceita dois formatos:
- **Multivariado** (recomendado): `[[f1,f2,f3,f4], ...]` - lista de timesteps com N features
- **Univariado**: `[1.2, 3.4, 5.6, ...]` - preço de fechamento apenas

**Exemplo de Requisição** (curl):
```bash
curl -X POST http://localhost:8000/infer/ \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": [
      [150.23, 152.10, 149.50, 1000000],
      [151.45, 153.20, 150.80, 1200000],
      [152.67, 154.40, 152.00, 950000]
    ]
  }'
```

**Exemplo de Resposta**:
```json
{
  "prediction": 153.45,
  "timestamp": "2024-01-09T14:45:30.123456",
  "latency_ms": 12.34
}
```

**Possíveis Erros**:
- `404`: Modelo ou scaler não encontrado
- `400`: Sequência inválida (dimensões incompatíveis)
- `500`: Erro durante inferência

---

#### **[GET] `/monitor/`**

Retorna métricas de saúde e monitoramento da API.

**Exemplo de Resposta**:
```json
{
  "status": "healthy",
  "total_inferences": 1250,
  "avg_latency_ms": 11.23,
  "min_latency_ms": 5.45,
  "max_latency_ms": 28.90,
  "last_inference_timestamp": "2024-01-09T14:50:15.654321"
}
```

---

#### **[GET] `/config/`**

Retorna a configuração do modelo campeão carregada em memória.

**Exemplo de Resposta**:
```json
{
  "metadata": {
    "selected_on": "2024-01-09 14:30:45",
    "experiment_name": "lstm_strategy_experiments",
    "run_id": "abc123...",
    "metric": "val_rmse",
    "metric_value": 2.12,
    "strategy": "NoProcessingMultipleStrategy"
  },
  "model": {
    "type": "LSTM",
    "input_size": 4,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "output_size": 1,
    "layer_config": ["LSTM", "Linear"]
  },
  "training": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 50,
    "shuffle": true
  },
  "data": {
    "tickers": ["AAPL", "MSFT"],
    "period": "1y",
    "seq_len": 20,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "scaler": "TimeSeriesScaler"
  }
}
```

---

### Teste Rápido dos Endpoints

```bash
# 1. Iniciar servidor em um terminal
uvicorn src.api.app:app --reload

# 2. Em outro terminal, testar endpoints
# Obter configuração
curl http://localhost:8000/config/ | python -m json.tool

# Fazer previsão
curl -X POST http://localhost:8000/infer/ \
  -H "Content-Type: application/json" \
  -d '{"sequence": [[150, 152, 149, 1000000], [151, 153, 150, 1200000]]}'

# Verificar saúde
curl http://localhost:8000/monitor/ | python -m json.tool
```

---

## 📊 Conceitos Técnicos

### DataPipeline e Padrão Strategy

O projeto utiliza o **padrão Strategy** para flexibilizar o processamento de dados:

#### **DataStrategy** (Interface Abstrata)

```
DataStrategy (ABC)
  ├─ NoProcessingSingle   → 1 ticker, sem engenharia
  ├─ NoProcessingMultiple → N tickers, sem engenharia
  ├─ RangeSingle          → 1 ticker + normalização Range
  └─ RangeMultiple        → N tickers + normalização Range
```

**Responsabilidades**:
1. Carregar dados históricos de ações
2. Validar colunas e tipos
3. Criar sequências temporais (sliding window)
4. Normalizar features (se aplicável)

**Exemplo**:
```python
from src.data.data import NoProcessingSingleStrategy
from src.data.scaler import TimeSeriesScaler

strategy = NoProcessingSingleStrategy()
X, y = strategy.process(
    tickers=["AAPL"],
    period="1y",
    seq_len=20
)
# X.shape = (samples, 20, 4)  # 4 features: High, Low, Close, Volume
# y.shape = (samples, 1)       # Target (preço de fechamento)
```

#### **DataPipeline** (Orchestrator)

Orquestra a estratégia escolhida e cria **DataLoaders**:

```python
from src.data.data import DataPipeline

pipeline = DataPipeline(
    strategy=NoProcessingSingleStrategy(),
    batch_size=32,
    shuffle=True,
    scaler=TimeSeriesScaler()
)

train_loader, val_loader, test_loader = pipeline.create_dataloaders()

for X_batch, y_batch in train_loader:
    # X_batch.shape = (32, 20, 4)
    # y_batch.shape = (32, 1)
    pass
```

---

### TrainerStrategy e TrainerContext

Padrão Strategy para orquestração de treinamento:

#### **TrainingStrategy** (Interface Abstrata)

Define QUAL pipeline, QUAL modelo e QUAIS hiperparâmetros usar:

```python
class TrainingStrategy(ABC):
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador único da estratégia"""
    
    @abstractmethod
    def get_data_pipeline(self) -> DataPipeline:
        """Retorna pipeline configurado"""
    
    @abstractmethod
    def get_model_factory(self, input_size: int) -> LSTMFactory:
        """Retorna factory com arquitetura configurada"""
    
    @abstractmethod
    def get_training_params(self) -> Dict[str, Any]:
        """Retorna hiperparâmetros de treinamento"""
```

**Estratégias Implementadas**:
- `NoProcessingSingleStrategy`: 1 ticker, dados brutos
- `NoProcessingMultipleStrategy`: N tickers, dados brutos
- `RangeSingleStrategy`: 1 ticker com normalização Range
- `RangeMultipleStrategy`: N tickers com normalização Range

#### **TrainerContext** (Executor)

Orquestra o treinamento:

```python
from src.training.trainer import TrainerContext

strategy = NoProcessingSingleStrategy(training_params)
trainer = TrainerContext(strategy)
metrics = trainer.train()

# Internamente:
# 1. Obtém pipeline da estratégia
# 2. Obtém model factory da estratégia
# 3. Cria LSTMLightningModule
# 4. Executa treinamento via PyTorch Lightning
# 5. Loga em MLflow
# 6. Salva modelo e artefatos
```

---

### Modelo LSTM e Factory

#### **LSTMFactory**

Cria arquiteturas LSTM flexíveis dado um `layer_config`:

```python
from src.models.lstm import LSTMFactory
from src.models.lstm_params import LSTMParams

params = LSTMParams(
    input_size=4,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    output_size=1
)

layer_config = ["LSTM", "Linear"]  # Sequência de camadas

factory = LSTMFactory(layer_config, params)
model = factory.create()

# Produz:
# LSTM(4 → 128, num_layers=2, dropout=0.2)
#   ↓
# Linear(128 → 1)
```

**Camadas Suportadas**:
- `LSTM`: Long Short-Term Memory
- `Linear`: Fully connected
- `ReLU`, `Tanh`, `Sigmoid`: Activation functions
- `Flatten`: Flatten tensor

---

### Normalização com TimeSeriesScaler

Scaler baseado em `StandardScaler` do scikit-learn, mantém histórico de fit:

```python
from src.data.scaler import TimeSeriesScaler

scaler = TimeSeriesScaler()

# Fit com dados de treino
train_data = ...  # shape: (n_samples, n_features)
scaler.fit(train_data)

# Transform dados
train_scaled = scaler.transform(train_data)

# Transform dados novos
new_data = ...
new_scaled = scaler.transform(new_data)

# Inverse transform
original = scaler.inverse_transform(new_scaled)
```

---

## ⭐ Modelo Campeão

### Como é Definido

1. **Execução**: `run_ray_experiments.py` testa múltiplas estratégias e hiperparâmetros
2. **Rastreamento**: MLflow registra cada run com métricas (val_rmse, val_mae, val_mape)
3. **Seleção**: `champion_selector.py` ordena por `val_rmse` e seleciona o menor
4. **Persistência**: Configuração é salva em `configs/best_config.yaml`

### Artefatos do Campeão

```
artifacts/
├── models/
│   └── model_final.pt           # Pesos do modelo (torch.save)
├── scalers/
│   └── scaler_final.pkl         # Scaler fitted (joblib.dump)
configs/
└── best_config.yaml             # Parâmetros e metadados
```

### Como é Utilizado

1. **API /train/**: Carrega `best_config.yaml` e treina modelo com esses parâmetros
2. **API /infer/**: Carrega modelo_final.pt e scaler_final.pkl para fazer predições
3. **Reprodutibilidade**: Qualquer pessoa pode recriar exatamente o mesmo modelo

---

## 🗓️ Roadmap Futuro

### 🔄 Implementação Futura - Endpoint de Update

Adicionar endpoint para retrainer o modelo campeão com dados novos:

```python
# [POST] /train/update
# Descição: Retrena modelo campeão com dados novos (sem busca de hiperparâmetros)

@router.post("/update")
def update_champion_model(request: UpdateRequest):
    """
    Retreina o modelo campeão com novos dados.
    
    Payload:
    {
        "period": "30d",  # Novo período
        "force_download": true  # Forçar download dos dados
    }
    
    Resposta:
    {
        "status": "success",
        "message": "Model updated with new data",
        "metrics": {...},
        "timestamp": "2024-01-09T..."
    }
    """
```

## 📖 Contributing

Para contribuir, consulte [CONTRIBUTING.md](CONTRIBUTING.md).

**Resumo rápido**:

1. Clone o repositório
2. Crie uma branch (`git checkout -b feature/minha-feature`)
3. Código + testes (`pytest`)
4. Formatação (`black .`)
5. Lint (`pylint src || true`)
6. Push e abra Pull Request

**Requisitos**:
- Python >= 3.13
- Cobertura de testes (happy path + 1 edge case mínimo)
- Black formatting
- Sem falhas críticas do Pylint

---

## 📚 Estrutura de Tipos

O projeto usa **dataclasses** para type safety:

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TrainingParams:
    tickers: List[str]
    period: Optional[str]
    seq_len: int
    num_epochs: int
    learning_rate: float
    batch_size: int
    layer_config: dict
    lstm_params: dict
    # ... mais parâmetros
```

---

## 🧪 Testes

Executar testes:

```bash
# Todos os testes
pytest -v

# Apenas teste do data pipeline
pytest -v src/data/tests/

# Com cobertura
pytest --cov=src

# Teste rápido para validar setup
python -m scripts.smoke_train
```

---

## 🔍 Monitoramento

### MLflow UI

```bash
mlflow server --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000
```

Acesse: **http://127.0.0.1:5000**

Visualize:
- Todos os experimentos executados
- Métricas de cada run
- Comparação de parâmetros
- Artefatos salvos

---

## 📋 Exemplo Completo de Workflow

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Baixar dados (se necessário)
python -m scripts.download_data

# 3. Rodar testes rápidos
python -m scripts.smoke_test_data_pipeline
python -m scripts.smoke_train

# 4. Otimizar hiperparâmetros
python -m scripts.run_ray_experiments
# ⏳ Aguarde conclusão (pode levar horas dependendo do dataset)

# 5. Selecionar modelo campeão
python -m scripts.champion_selector

# 6. Iniciar API
uvicorn src.api.app:app --reload

# 7. (Novo terminal) Visualizar experimentos
mlflow server --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000

# 8. Fazer previsões via API
curl -X POST http://localhost:8000/infer/ \
  -H "Content-Type: application/json" \
  -d '{"sequence": [[150, 152, 149, 1000000], [151, 153, 150, 1200000]]}'
```

---

## 📝 Licença

Projeto desenvolvido para **Tech Challenge 4** da Pós Graduação em Machine Learning Engineering.

---

## 📞 Suporte

Para dúvidas ou issues, abra uma issue no repositório ou consulte [CONTRIBUTING.md](CONTRIBUTING.md).