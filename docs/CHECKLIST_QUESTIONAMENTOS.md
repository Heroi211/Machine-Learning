# Checklist — questionamentos da jornada de testes

Itens levantados para decisão de produto/arquitetura e eventual implementação.  
Marcar `- [x]` quando estiver resolvido ou descartado.

---

## 1. `objective` como enum no Swagger

- [x] **Objetivo:** expor no OpenAPI apenas valores permitidos (ex.: `heart_disease`, futuros domínios), evitando strings livres inválidas.
- [x] **Notas:** enum único `MLDomain` em `schemas/processor_schemas.py` (`heart_disease`, `churn`; novos valores = novo membro + strategy em `STRATEGY_REGISTRY` quando for FE). Rotas: `objective`/`domain` em `Form`, path `GET .../deployments/{domain}/history`, bodies `PromoteRequest` / `RollbackRequest`. Swagger/OpenAPI gera `components.schemas.MLDomain` com `enum` — dropdown no UI. **Predict:** `PredictRequest.domain` também é `MLDomain` (`PredictDomain` permanece alias).

---

## 2. Rotas síncronas Baseline / FE vs treino só pelo Airflow

- [x] **Pergunta:** se o treino “oficial” for **só via Airflow**, as rotas `POST .../admin/train/baseline` e `.../feature-engineering` ainda são necessárias?
- [x] **Opções:** remover; manter só em **dev** (flag `DEBUG` / env); manter sempre para **depuração linha a linha** e testes rápidos.
- [x] **Decisão registada:** manter as rotas síncronas para **desenvolvimento e depuração linha a linha** (testes rápidos, breakpoints). Em **produção**, baseline/FE síncronos e **trigger pela API** respondem **403**; o treino em prd é **Airflow pela UI** (Variables + CSV no volume + Trigger DAG). Em **não produção**, a API pode usar **`POST .../admin/train/trigger-dag`**. Implementação: `Settings.is_production`, `require_sync_training_routes_enabled`, `require_airflow_api_trigger_enabled` em `core/deps.py`.

---

## 3. `load_dotenv()` no Baseline vs `settings` (`core.configs`)

- [ ] **Pergunta:** Baseline (ou outro módulo) chama `load_dotenv()` mas o projeto centraliza env em `Settings` — é redundante?
- [ ] **Notas:** em container, variáveis vêm do Compose; `load_dotenv()` ajuda só em execução local sem Compose. Avaliar remover ou restringir a `if __main__` / dev.

---

## 4. `PipelineRuns.active` — quando fica `true`

- [x] **Ideia:** criação com **default `True`** (`modelsGeneric`); em **falha** de treino → `active=False`; sucesso mantém `True`. Quem “filtra” readiness real é **`status == completed`** + artefactos na promoção.
- [x] **Impacto:** queries que filtram `active` em `PipelineRuns` / relatórios — **promote** (`deployment_service.promote_pipeline_run`) exige `PipelineRuns.active == True` **e** `status == completed` **e** ficheiro de modelo existente.

---

## 5. `PipelineRuns` — `error_message` implica `active = false`

- [x] **Regra:** se `error_message` estiver preenchido (ou `status == failed`), `active` deve ser **`false`** automaticamente (constraint lógica ou validação ao persistir).
- [x] **Notas:** `PipelineRuns` usa o `active` de `modelsGeneric` (**default `True`** na criação). Em falha de treino: `run.active = False` em `processor_service` (baseline/FE). Promoção continua a exigir `status == completed` + modelo existente, por isso runs em `processing` não são promovíveis mesmo com `active=True`. **Brecha conhecida:** o DAG Airflow (`ml_training_pipeline`) ainda **não persiste** `PipelineRuns` na BD — só as rotas síncronas criam linhas; alinhar DAG↔BD num passo futuro se o treino “oficial” tiver de aparecer em `pipeline_runs`.

---

## 6. Dozzle — visibilidade API vs ML

- [ ] **Problema:** Dozzle mostra sobretudo **logs de contentores**; fluxo **ML** (pipelines, stdout dos jobs) pode não aparecer como desejado na API.
- [ ] **Opções:** volumes de logs partilhados; logs estruturados na API; documentar que treino pesado corre no worker Airflow e abrir logs desse serviço; Grafana/Loki (fora de escopo imediato).
- [ ] **Decisão registada:** _…_

---

## 7. Payload do `predict` validável no schema

- [x] **Objetivo:** deixar de usar `features: dict` genérico; usar modelo Pydantic **por domínio** ou `TypedDict` / schema JSON com campos e tipos (ex.: `heart_disease` com `age`, `chol`, …).
- [x] **Trade-off:** um schema por domínio aumenta manutenção; alternativa: `features` com validação condicional (`discriminator` por `domain`) ou endpoint por domínio — **decisão atual:** um modelo `HeartDiseaseFeaturesInput` + `PredictRequest` (`domain` + `features`); `extra="forbid"` em ambos.
- [x] **Notas:** implementado em `schemas/processor_schemas.py` (`PredictDomain`, `HeartDiseaseFeaturesInput`, `PredictRequest`). Rota `POST /predict` em `api/v1/endpoints/processor.py` faz `model_dump(mode="json", by_alias=True)` para preservar chaves com espaço/hífen no JSON. **Próximo passo (outros domínios):** `Union` discriminada por `domain` ou schemas separados.

---

## Ordem sugerida de ataque (referência)

1. Schema do **predict** + **enum** `MLDomain` / `objective` nas rotas (feito — ver §1 e §7).  
2. Regras **`active` / `error_message`** em `PipelineRuns` (feito — ver §4 e §5).  
3. **Rotas síncronas** vs só Airflow (feito — ver §2: bloqueio em prd + trigger-dag).  
4. **load_dotenv** / limpeza de env.  
5. **Dozzle** / observabilidade ML (documentação primeiro, ferramenta depois).

---

*Gerado a partir dos questionamentos da jornada de testes — atualizar conforme forem decididos.*
