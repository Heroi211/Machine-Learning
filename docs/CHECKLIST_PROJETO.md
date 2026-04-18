# Checklist do Projeto — Machine Learning Engineering

Escopo definido: **classificação binária tabulada** (heart disease e churn).
Regressão, séries temporais e problemas multiclasse estão fora do escopo.

Legenda: `[ ]` pendente · `[x]` concluído · **P0** crítico · **P1** importante · **P2** evolução

---

## Bloco 1 — Qualidade e corretude dos pipelines

### Baseline (`services/pipelines/baseline.py`)

**Contrato de entrada definido:** o CSV de baseline deve ter a coluna `target` sempre como **última coluna** e **sem coluna de ID**. Isso é uma convenção do projeto, não um bug.

- [x] **P0** Documentar o contrato de entrada do CSV na docstring da classe e no README: "última coluna = target, sem coluna de ID".
- [x] **P0** Documentar explicitamente na docstring da classe que o Baseline suporta apenas **classificação binária**.
- [x] **P1** Substituir labels hardcoded `f'Sem {self.objective}'` nos gráficos por parâmetro `class_labels` — quebra semanticamente em churn (ex.: "Sem Churn" vs "Churn" em vez de "Sem heart_disease").
- [ ] **P1** Formalizar a dívida técnica de desbalanceamento: remover o comentário solto do código e registrar como decisão documentada no README (ex.: "usando `class_weight='balanced'` como mitigação conhecida").

### Feature Engineering (`services/pipelines/feature_engineering.py`)

- [x] **P0** Substituir `assert "target" in df.columns` por `if "target" not in df.columns: raise ValueError(...)`.
- [x] **P1** Tornar `time_limit_minutes` parâmetro do método `run()` — hoje hardcoded como `2`, insuficiente para tuning real.
- [x] **P1** Tornar `acc_target` parâmetro do método `run()` — hoje hardcoded como `0.90`.

### Suporte a múltiplos domínios (classificação binária)

- [ ] **P1** Implementar `ChurnFeatures` — segunda `FeatureStrategy` com dataset público de churn (ex.: Telco Customer Churn do Kaggle), demonstrando o padrão de extensão do registry.
- [ ] **P1** Registar `churn` no `STRATEGY_REGISTRY` em `feature_strategies/__init__.py`.
- [ ] **P1** Validar que o fluxo completo (upload → baseline → FE → promote → predict) funciona para `churn` sem alteração nos pipelines centrais.

---

## Bloco 2 — API de serving

- [x] **P1** Rota `GET /processor/admin/deployments/{domain}/history` — lista versões promovidas para o domínio com métricas snapshot, para comparar modelos antes de promover.
- [x] **P1** Endpoint de rollback `POST /processor/admin/rollback` — reativa o deployment `archived` imediatamente anterior do domínio.
- [ ] **P2** Rota `GET /processor/admin/runs` — listar `pipeline_runs` por domínio com status e métricas (facilita escolher qual run promover sem precisar consultar o banco diretamente).

---

## Bloco 3 — Orquestração com Airflow

- [x] **P1** Criar `docker-compose.yml` com todos os serviços: `api` + `postgres` + `airflow` + `mlflow` com volumes compartilhados entre containers.
- [x] **P1** Definir DAG `ml_training_pipeline` com tasks sequenciais:
  - `task_validate_input` — valida schema e colunas mínimas do CSV para o `objective`.
  - `task_run_baseline` — executa `Baseline` e persiste `pipeline_run` no banco.
  - `task_run_fe` — executa `FeatureEngineering` e persiste `pipeline_run` no banco.
  - `task_notify_complete` — loga conclusão com `run_id` para o admin decidir sobre promoção.
- [x] **P1** Endpoint na API `POST /processor/admin/train/trigger-dag` — grava CSV em volume compartilhado e dispara o DAG via Airflow REST API (desacopla treino da thread HTTP).
- [x] **P2** Parametrizar DAG com `objective` e `optimization_metric` via `dag_run.conf` para suportar heart disease e churn no mesmo DAG.
- [x] **P2** Task de notificação ao admin quando o DAG conclui (log estruturado ou chamada a endpoint interno).

---

## Bloco 4 — Observabilidade de logs

- [x] **P1** Garantir que API e pipelines emitem logs para **stdout** (além dos arquivos) — necessário para `docker logs` e Dozzle funcionarem.
- [x] **P1** Adicionar serviço **Dozzle** no `docker-compose.yml` (`image: amir20/dozzle`, porta `8888`) — UI web para visualização de logs de todos os containers sem configuração adicional.
- [ ] **P2** Separar `logs/` como volume nomeado no compose para persistência entre reinicializações do container.
- [ ] **P2** Script ou notebook leve que lê `access.jsonl` e exibe tabela de latência por rota e taxa de erros — para uso em demo ou relatório.

---

## Bloco 5 — Scripts de manutenção

### Latência (`scripts/maintenance/latency_report.py`)

- [x] **P0** Adicionar filtragem por rota: separar métricas de `/predict` das demais no CSV de saída — sem isso, a latência do predict fica mascarada por rotas de treino.
- [x] **P1** Adicionar coluna `slo_status` no CSV: `ok` se p95 ≤ threshold configurável, `breach` se exceder.
- [x] **P1** Adicionar taxa de erro (% de respostas 4xx + 5xx) ao relatório.
- [x] **P1** Documentar no README o SLO do projeto (ex.: "p95 de `/predict` < 300ms").
- [ ] **P2** Campo `delta_p95_ms` comparando com o relatório anterior — detectar degradação progressiva.

### Drift (`scripts/maintenance/drift_report.py`)

- [x] **P0** Adicionar coluna `status` no CSV de saída: `ok` (PSI < 0.10) · `warning` (0.10–0.25) · `critical` (> 0.25) — tornar o relatório acionável sem leitura manual dos números.
- [x] **P1** Adicionar guarda para features com poucos valores únicos: se `len(breakpoints) < 3` após `np.unique`, pular a feature com log de aviso — evita PSI instável em binárias (relevante para churn).
- [x] **P1** Documentar periodicidade e processo de decisão no README: "PSI critical → abrir retreino → promover novo run → reavaliar drift".
- [x] **P2** Linha de resumo no CSV: PSI médio geral, contagem de features por status — leitura rápida sem abrir o arquivo completo.

---

## Bloco 6 — Versionamento e promoção de modelos

- [x] **P1** Documentar critério formal de promoção no README: "promover quando métrica principal do novo run supera o ativo em ≥ 2% E PSI médio das features < 0.10".
- [x] **P1** Implementar `GET /processor/admin/deployments/{domain}/history` (ver Bloco 2).
- [x] **P1** Implementar endpoint de rollback (ver Bloco 2).
- [ ] **P2** Campo `reason` opcional no `POST /promote` — texto livre que documenta o motivo da promoção para auditoria.
- [ ] **P2** Confirmar que o índice único `uq_deployed_models_one_active_per_domain` existe na base de dados em uso (presente no `init_db/database.sql`, verificar base de produção).

---

## Bloco 7 — Entrega e documentação (TC)

- [x] **P0** `docker-compose.yml` funcional: `git clone` → `docker compose up` → API respondendo em `/docs`.
- [x] **P0** Arquivo `.env.example` com todas as variáveis necessárias e sem valores reais.
- [x] **P0** README com passo a passo completo: setup → login → upload CSV → train → promote → predict.
- [x] **P1** EDA documentada: seção no relatório com os gráficos gerados pelo Baseline (missings, target, outliers) e interpretação para cada domínio (heart disease + churn).
- [x] **P1** Tabela de comparação no relatório: **Baseline vs Feature Engineering** — métricas lado a lado com interpretação.
- [x] **P1** Diagrama do fluxo completo: dados → Airflow (treino) → MLflow (registro) → promoção → API serving → monitoramento.
- [x] **P1** Seção de **limitações e trabalho futuro**: escopo binário, tuning limitado, drift offline, ausência de feature store formal.
- [ ] **P2** Demo gravada ou passo a passo reprodutível cobrindo o fluxo de administrador e consumidor para os dois domínios.
