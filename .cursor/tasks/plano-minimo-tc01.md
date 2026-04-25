# Plano Mínimo — Tech Challenge Fase 01 (TC_01)

## Objetivo

- Entregar o menor conjunto de implementações que maximize a pontuação esperada do Tech Challenge Fase 01 (MLP PyTorch para churn + pipeline end-to-end).
- Separar claramente "fazer agora" de "adiar aceitando penalização".
- Governar a execução via agentes especializados (`ml-agent`, `api-agent`, `mlops-agent`) já definidos em `.cursor/agents/`.

## Contexto

- Enunciado oficial: `reference/TC_01.pdf` (Tema Central: Rede Neural para Previsão de Churn com Pipeline Profissional End-to-End).
- Critérios e pesos oficiais:
  - Qualidade de código e estrutura — 20 %.
  - Rede neural PyTorch (MLP + early stopping + baselines) — 25 %.
  - Pipeline e reprodutibilidade (sklearn, seeds, `pyproject.toml`, instala do zero) — 15 %.
  - API de inferência (FastAPI, Pydantic, logging, testes) — 15 %.
  - Documentação e Model Card — 10 %.
  - Vídeo STAR (5 min) — 10 %.
  - Bônus deploy em nuvem — 5 %.
- Estado atual (resumo, ver `docs/CHECKLIST_TECH_CHALLENGE_FASE01.md`):
  - **Tem**: FastAPI `POST /predict`, Pydantic `extra="forbid"`, `Baseline` + `FeatureEngineering` sklearn, MLflow tracking, `StratifiedKFold`, seeds, logging estruturado + middleware latência, EDA automatizada, docs amplas, promote/rollback de modelo, Airflow, Postgres, JWT/RBAC.
  - **Parcial**: baseline (falta `DummyClassifier`), análise custo FP/FN, playbook de monitoramento, enum churn sem `FeatureStrategy`.
  - **Falta**: MLP PyTorch, `pyproject.toml`/ruff, `tests/` com pytest, pandera, Makefile, `/health`, Model Card, doc de arquitetura de deploy, vídeo STAR, dataset Telco carregado.

## Escopo e Restrições

- Pode alterar: `services/pipelines/`, `schemas/`, `api/`, `docs/`, `scripts/`, raiz (novos `pyproject.toml`, `Makefile`, `tests/`, `CHANGELOG.md`, `CONTRIBUTING.md`).
- Não deve alterar: regras de autenticação/RBAC, `docker-compose.yaml` em estrutura, DAG Airflow em produção, contratos de schemas já expostos (apenas extensões aditivas).
- Nenhuma mudança destrutiva em artefatos persistidos; modelos atuais (sklearn) permanecem servidos em produção. MLP entra como módulo de benchmark/experimentação.

## Critério de Pronto

- `python scripts/train_mlp_benchmark.py` executa do início ao fim e gera tabela comparativa + learning curve.
- `pytest` verde localmente com ≥ 3 testes (smoke, schema, API).
- `ruff check .` sem erros.
- `make lint test run` funcionais.
- `pip install -e .` (via `pyproject.toml`) instala o projeto do zero.
- `GET /health` responde 200 sem autenticação.
- `docs/MODEL_CARD.md`, `docs/MLP_COMPARATIVO.md`, `docs/ARQUITETURA_DEPLOY.md`, `docs/PLANO_MONITORAMENTO.md` presentes e linkados no README.
- Vídeo STAR de 5 min gravado (`docs/ROTEIRO_VIDEO_STAR.md` consolida o roteiro).
- README com instruções de setup + execução + arquitetura + link para vídeo.

## Métricas-Alvo

- Pontuação esperada ≥ 85/100 (sem bônus de nuvem).
- Cobertura dos critérios: 100 % obrigatórios "Fazer agora" + ≥ 70 % dos "Fazer se sobrar capacidade".
- MLP: treinar até convergir (early stopping `patience=15`) e reportar ≥ 5 métricas (accuracy, precision, recall, F1, ROC-AUC) comparando com ≥ 4 baselines (Dummy, Logistic Regression, Random Forest, Gradient Boosting / XGBoost).

## Plano de Execução

Dividido por fases. Cada item cita o agente responsável.

### Fase 0 — Alinhamento e contratos (0,5 dia)

1. Consolidar dataset Telco em `data/raw/telco_customer_churn.csv` — `ml-agent`.
2. Decidir schema do payload `/predict` para churn e atualizar `PredictRequest` discriminado por `domain` — `api-agent`.
3. Criar esqueleto `pyproject.toml` + `Makefile` (sem remover `requirements.txt` ainda) — `mlops-agent`.
4. Abrir `tests/` com um `test_smoke.py` mínimo — `api-agent`.

**Saída**: contratos e esqueletos prontos; nada ainda funcional.

### Fase 1 — MVP funcional fim-a-fim (1,5 dia)

1. `services/pipelines/mlp_torch.py` (modelo + `train_mlp` com batching + early stopping + `BCEWithLogitsLoss` com `pos_weight`) — `ml-agent`.
2. `services/pipelines/feature_strategies/churn_features.py` + registrar em `STRATEGY_REGISTRY` — `ml-agent`.
3. `scripts/train_mlp_benchmark.py` (roda sobre o split churn, compara MLP vs 4 baselines, gera CSV + figura) — `ml-agent`.
4. `api/health.py` com `/health` e `/health/ready` + registro em `main.py` — `api-agent`.
5. `tests/` com smoke, schema Pydantic, teste de API para `/predict` (cliente feliz) — `api-agent`.
6. `pyproject.toml` como *single source of truth*: dependências, config `ruff`, config `pytest` — `mlops-agent`.

**Saída**: MLP treinado e avaliado; API tem health e testes; projeto instala via `pip install -e .`.

### Fase 2 — Robustez mínima para entrega (1 dia)

1. `docs/MODEL_CARD.md` (dataset, métricas, limitações, vieses, cenários de falha) — `ml-agent`.
2. `docs/MLP_COMPARATIVO.md` (tabela + learning curve + discussão) — `ml-agent`.
3. `docs/ARQUITETURA_DEPLOY.md` (batch vs real-time + justificativa) — `mlops-agent`.
4. `docs/PLANO_MONITORAMENTO.md` (métricas, alertas, playbook de resposta; reaproveitar scripts de drift/latência já existentes) — `mlops-agent`.
5. README: reorganizar para setup → execução → arquitetura → links para Model Card/MLP/Deploy/Monitoramento/Vídeo — `mlops-agent`.
6. Gravar vídeo STAR de 5 min (roteiro + gravação) — responsabilidade do grupo, guia em `docs/ROTEIRO_VIDEO_STAR.md`.

**Saída**: entrega completa com documentação, testes e vídeo; MLflow já rastreia o MLP como run separado.

### Fase 3 — Opcional (bônus 5 %)

- Deploy da API em AWS (ECS Fargate) ou equivalente — `mlops-agent`. Só iniciar se Fase 2 concluída.

## Agentes e Responsabilidades

| Domínio | Agente      | Itens principais                                                          |
| ------- | ----------- | ------------------------------------------------------------------------- |
| ML      | `ml-agent`  | MLP PyTorch, `ChurnFeatures`, benchmark, Model Card, MLP_COMPARATIVO.     |
| API     | `api-agent` | `/health`, schema churn em `PredictRequest`, testes de API, validação.    |
| MLOps   | `mlops-agent` | `pyproject.toml`, ruff, Makefile, arquitetura de deploy, monitoramento, deploy cloud opcional. |

## Riscos e Mitigações

- **Risco**: MLP não converge ou empata com baselines no dataset pequeno de heart; **mitigação**: usar Telco (~7 000 linhas) como dataset principal do benchmark; documentar honestamente.
- **Risco**: migração `requirements.txt` → `pyproject.toml` quebra Docker; **mitigação**: manter ambos até validar build da imagem; remover `requirements.txt` só após `docker compose build` passar.
- **Risco**: `torch` CPU pesa ~200 MB no container; **mitigação**: instalar só no venv local do dev (MLP é experimentação, não servido pela API).
- **Risco**: vídeo STAR é entrega não-código fácil de esquecer; **mitigação**: reservar bloco de tempo dedicado na Fase 2 e deixar roteiro pronto antes de gravar.
- **Risco**: Pandera para pipeline sklearn custom consome tempo sem retorno proporcional; **mitigação**: entregar validação apenas no endpoint (`PredictRequest`) e via teste de schema — pandera completo fica como "adiar e aceitar penalização".

## Referências Cruzadas

- `reference/TC_01.pdf`
- `docs/CHECKLIST_TECH_CHALLENGE_FASE01.md`
- `docs/PLANO_MINIMO_MLP_PYTORCH.md`
- `docs/CHECKLIST_BUILD_CHURN_FEATURES.md`
- `docs/ESBOCO_HEALTH_CHANGELOG.md`
- `docs/RELATORIO_PYPROJECT_RUFF_MAKEFILE.md`

## Premissas e Confiança

- (Alta) Pesos de avaliação e entregas obrigatórias são literalmente os do PDF (Seção *Critérios de Avaliação*).
- (Alta) Projeto atual já cumpre a maioria dos itens de API/MLOps/MLflow conforme `docs/CHECKLIST_TECH_CHALLENGE_FASE01.md`.
- (Média) Dataset alvo é Telco IBM Customer Churn (o PDF sugere e é o padrão de mercado; o usuário pode optar por outro ≥ 5 000 registros e ≥ 10 features).
- (Média) MLP não precisa ser servido pela API — leitura literal do PDF: ele pede MLP *treinado e comparado*, não *em produção*.
- (Baixa) Nota-alvo 85/100 é estimativa; depende da qualidade do vídeo STAR e da documentação narrativa.
