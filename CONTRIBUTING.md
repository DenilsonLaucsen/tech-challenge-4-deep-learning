# Contributing to Tech Challenge Deep Learning

Obrigado por considerar contribuir para este projeto! Este é um **micro-framework para produtização de modelos LSTM**, focado em treinar, servir e monitorar modelos com rapidez e escalabilidade.

## Sumário

- [Código de Conduta](#código-de-conduta)
- [Como Começar](#como-começar)
- [Padrões de Código](#padrões-de-código)
- [Testes](#testes)
- [Branches e Pull Requests](#branches-e-pull-requests)
- [Stack Obrigatória](#stack-obrigatória)
- [Endpoints](#endpoints)
- [CI/CD e Segurança](#cicd-e-segurança)
- [ADR](#adr)
- [Dúvidas?](#dúvidas)
- [TODO](#todo)

---

## Código de Conduta

Seguimos o [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Ao participar, você concorda em manter um ambiente respeitoso e inclusivo para todos.

---

## Como Começar

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

# Rode testes para validar setup
pytest -q
```

---

## Padrões de Código

### Formatação e Lint

**Antes de cada commit, execute:**

```bash
# Formata o código
black .

# Verifica lint (falhas não bloqueiam, apenas informam)
pylint src || true
```

- **Formatador:** [Black](https://black.readthedocs.io/)
- **Linter:** [Pylint](https://pylint.org/)
- **Python:** >= 3.13 obrigatório

### Dependências

- **Nunca** instale pacotes com `pip install` diretamente.
- Adicione dependências apenas em `pyproject.toml` ou `requirements.txt`.

---

## Testes

### Requisitos Mínimos

- **PyTest** é obrigatório.
- Cada novo módulo/função deve incluir:
  - ✅ **Happy path:** teste com entrada válida.
  - ✅ **1 edge case:** teste de limite, erro ou caso atípico.

### Exemplo

```bash
pytest -q              # Modo quiet
pytest -v src/models  # Verbose num diretório específico
pytest --cov src      # Com cobertura
```

---

## Branches e Pull Requests

### Padrão Trunk-Based

- **Branch principal:** `main` (trunk).
- **Branches de trabalho:** baseadas em `main`.

### Nomes de Branches

```
feat/<descricao-curta>      # Nova feature
fix/<descricao-curta>       # Bug fix
chore/<descricao-curta>     # Manutenção, deps, etc.
hotfix/<descricao-curta>    # Fix urgente em prod
```

### Checklist Antes do PR

- [ ] Código formatado com `black .`
- [ ] Lint revisado com `pylint src || true`
- [ ] Testes passando: `pytest -q`
- [ ] Documentação atualizada (docstrings, README se aplicável)
- [ ] Issue referenciada no título ou body do PR
- [ ] 1 revisor técnico + 1 revisor ML (se aplicável)

### Comandos Típicos

```bash
black .
pylint src || true
pytest -q
git push origin feat/sua-feature
```

---

## Stack Obrigatória

- **PyTorch:** framework de deep learning.
- **PyTorch Lightning:** abstração high-level para treino.
- **Ray:** HPO e distribuição.
- **MLflow:** tracking de experimentos e modelos.
- **FastAPI:** endpoints HTTP.

---

## Estrutura Mínima de Pastas

```
tech-challenge-deep-learning/
├── src/
│   ├── api/           # FastAPI app
│   ├── data/          # Data loaders e prep
│   ├── models/        # Arquitetura LSTM
│   ├── training/      # Treino com Lightning
│   └── utils/         # Utilitários
├── tests/             # Testes (PyTest)
├── requirements.txt   # Dependências
└── CONTRIBUTING.md    # Este arquivo
```

---

## Endpoints

Endpoints esperados:

- **`POST /train`** – Dispara treino (com params via JSON).
- **`POST /infer`** – Inferência em batch.
- **`POST /update`** – Atualiza modelo registrado.
- **`GET /monitor`** – Status e métricas.
- **`GET /config`** – Config atual.

Cada endpoint deve ser justificado no `README.md` com exemplos de request/response.

---

## CI/CD e Segurança

- **Workflows:** rodados em cada PR.
- **Deploy:** requer aprovação manual; sem auto-deploy.
- **Secrets:** **nunca** commite secrets. Use variáveis de ambiente ou secrets manager.
- **Credenciais:** sempre adicione à `.gitignore` e documente em `.env.example`.

---

## ADR

**Architecture Decision Records** documentam decisões técnicas importantes.

- **Localização:** `docs/adr/`
- **Nomeação:** `ADR-001-descricao.md`, `ADR-002-...`, etc.
- **Template:** incluir contexto, opções consideradas, decisão e consequências.

---

## Dúvidas?

- **Features ou bugs:** abra uma [issue](../../issues) com rótulo `feature` ou `bug`.
- **Perguntas gerais:** abra issue com rótulo `question`.
- **Discussões técnicas:** use [Discussions](../../discussions) ou canal interno.

---

## TODO

Itens a implementar no repositório:

- [ ] Adicionar template de PR (`.github/pull_request_template.md`)
- [ ] Adicionar templates de ISSUE (`bug.md`, `feature.md`, `question.md`)
- [ ] Script de setup automático (`scripts/setup.sh`)
- [ ] Exemplo de CI (GitHub Actions ou GitLab CI)
- [ ] Documentação de deployment
- [ ] Código de exemplo completo de treino (notebook ou script)
