# Desenho do sistema — pipeline ML + monitorização

## Versão gráfica (PNG / SVG)

Imagens geradas em `docs/` (renderizadas a partir do Mermaid abaixo via `@mermaid-js/mermaid-cli`):

| Diagrama | PNG | SVG |
|----------|-----|-----|
| Arquitetura completa | [`docs/arquitetura_ml.png`](arquitetura_ml.png) | [`docs/arquitetura_ml.svg`](arquitetura_ml.svg) |
| Zoom DAGs Airflow | [`docs/arquitetura_dags.png`](arquitetura_dags.png) | [`docs/arquitetura_dags.svg`](arquitetura_dags.svg) |

![Arquitetura do sistema](arquitetura_ml.svg)

### Como regenerar

```bash
mkdir -p docs
npx --yes -p @mermaid-js/mermaid-cli mmdc \
  -i desenho.md -o docs/arquitetura_ml.png \
  -t neutral -b transparent -w 1800 -H 1400
```

> Em WSL/Linux sem sandbox do Chrome, anexar `-p puppeteer.json` com
> `{"args":["--no-sandbox","--disable-setuid-sandbox"]}`.

Alternativa visual: abrir **[mermaid.live](https://mermaid.live)** → colar o bloco
` ```mermaid ` abaixo → **Actions → PNG/SVG**.

---

## Diagrama em código (Mermaid)

O mesmo diagrama — útil para GitHub, Cursor e revisões em diff:

```mermaid
flowchart TB
  subgraph actores ["Actores"]
    CLI["Cliente / App"]
    OPS["Operador ML"]
  end

  subgraph ingress ["Entrada"]
    CLI -->|HTTPS JWT| API
    OPS -->|Admin train / predict / promote| API
    OPS -->|Airflow UI / CLI| AF_UI["Airflow"]
  end

  subgraph api_svc ["API FastAPI"]
    API["Rotas v1: auth, processor, health"]
    MW["Middleware → access.jsonl<br/>duration_ms, path, status"]
    API --- MW
  end

  subgraph airflow_svc ["Airflow"]
    AF_UI --> SCH["Scheduler"]
    SCH --> DAG_TRAIN["DAG ml_training_pipeline<br/>Baseline → FE → promote opcional"]
    SCH --> DAG_DRIFT["DAG ml_drift_monitoring<br/>export_predictions → run_drift_report"]
  end

  subgraph persistencia ["Persistência e artefactos"]
    PG[("PostgreSQL<br/>users · pipeline_runs<br/>predictions · deployed_models")]
    ML_API["MLflow API<br/>mlflow.db · mlruns/"]
    ML_AF["MLflow Airflow<br/>airflow_mlflow.db<br/>airflow_store/"]
    VOL["Volume ml_shared + binds<br/>src/data · models · mlruns · uploads"]
  end

  subgraph pipelines ["Pipelines ML"]
    BL["Baseline"]
    FE["Feature Engineering"]
    INF["Inferência sklearn / MLP"]
  end

  subgraph manutencao ["Manutenção"]
    LOGS["PATH_API_REQUEST_LOGS<br/>access.jsonl*"]
    LAT["latency_report.py"]
    EXP["export_predictions<br/>(task Airflow)"]
    DRIFT_PY["drift_report.py"]
    OUT["src/artifacts/reports<br/>latency_*.csv · drift_psi_*.csv"]
  end

  API --> PG
  API --> VOL
  API --> INF
  INF --> PG

  DAG_TRAIN --> BL
  DAG_TRAIN --> FE
  BL --> VOL
  FE --> VOL
  BL --> ML_AF
  FE --> ML_AF
  OPS -. treino manual .-> API
  API -. FE/baseline .-> ML_API

  PG --> EXP
  DAG_DRIFT --> EXP
  EXP --> DRIFT_PY
  VOL -. CSV referência PSI<br/>baseline_sample* .-> DRIFT_PY
  DRIFT_PY --> OUT

  MW -.-> LOGS
  LOGS --> LAT
  LAT --> OUT

  style manutencao fill:#f5f5f5,stroke:#666
  style persistencia fill:#eef6ff,stroke:#3366cc
  style api_svc fill:#fff8e6,stroke:#cc9900
  style airflow_svc fill:#f0fff0,stroke:#339933
```

## Legenda

| Estilo | Significado |
|--------|-------------|
| Linha cheia | Fluxo principal de dados / treino / inferência |
| Linha tracejada | Treino manual na API; logs → latência; CSV de referência para PSI |
| Verde | Orquestração Airflow (**dois** DAGs independentes) |
| Amarelo | API e telemetria de latência |
| Azul | BD, volumes, MLflow |
| Cinzento | Relatórios offline |

## Zoom só nas DAGs Airflow

![Zoom DAGs Airflow](arquitetura_dags.svg)

```mermaid
flowchart LR
  subgraph train ["ml_training_pipeline"]
    V["validate_input"]
    D["deactivate_manual_runs"]
    B["run_baseline"]
    F["run_fe"]
    P["promote_fe_optional"]
    N["notify_complete"]
    V --> D --> B --> F --> P --> N
  end

  subgraph drift ["ml_drift_monitoring"]
    E["export_predictions"]
    R["run_drift_report"]
    E --> R
  end

  PG[(predictions)]
  REP["reports/ drift_psi_*.csv"]

  PG -.-> E
  R --> REP
```
