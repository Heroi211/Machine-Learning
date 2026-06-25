# Migração para arquitectura `_ring`

Checklist por fase. Estado inicial: **Fase 0 concluída** (documentação + pastas + verificador).

---

## Fase 0 — Acordo e documentação ✅

- [x] `docs/PLATFORM_CONTRACT.md` — fluxo imutável train → promote → predict
- [x] `docs/ARCHITECTURE_RINGS.md` — anéis e diagramas
- [x] `docs/IMPORT_RULES_RINGS.md` — matriz de imports
- [x] Pastas `src/*_ring/` com README por anel
- [x] `scripts/check_ring_imports.py` — verificador de imports
- [ ] Review em equipa / banca (marcar data)

---

## Fase 1 — `ml_core_ring` + registries ✅

- [x] Mover `src/core/ml/` → `src/ml_core_ring/` (aliases temporários em `core/ml`)
- [x] `train_backend.py` + `TRAIN_BACKEND_REGISTRY`
- [x] `orchestration_hooks.py` (Airflow → backend)
- [x] Testes unitários registries (`tests/ml_core_ring/`)

---

## Fase 2 — `executors_ring` ✅

- [x] `executors_ring/tabular_classification/` wrap Baseline + FE + strategies + MLP inference
- [x] `executors_ring/recommendation/` wrap runner + worker stub
- [x] Backends registados no `TRAIN_BACKEND_REGISTRY` (via `import executors_ring`)
- [x] `TorchBundleEngine` movido para executors (ml_core deixa de importar pipelines)

---

## Fase 3 — `orchestration_ring` (Airflow central) ✅

- [x] DAG `ml_training_dispatch` com `domain` no conf
- [x] Tasks tabular + recommendation (registry → executors)
- [x] `persist_run` partilhada → `pipeline_runs` (+ recomendação)
- [x] `ml_training_pipeline` delega a `orchestration_ring.tabular_training`

---

## Fase 4 — `platform_ring` ✅

- [x] Serviços `platform_ring/` (trigger, promote, runs) — API delega
- [x] `trigger-dag` genérico por `domain` → `ml_training_dispatch`
- [x] Promote generalizado (tabular FE + recommendation)
- [x] `/predict` recommendation (`recommended_items` + engine `recommendation_torch`)

---

## Fase 5 — Worker recommendation ✅

- [x] Container `worker-recommendation` (`POST /train`, `GET /health`)
- [x] Compose integrado (`nwprocessing`: MLflow + worker + volumes reco)
- [x] Airflow delega via `WORKER_RECOMMENDATION_URL`
- [x] `ML_PROJECT_ROOT` unifica paths reco (host ↔ contentor)
- [x] `docker-compose.tc02.yml` deprecado — stack principal

---

## Fase 5.5 — Rotas por domínio + paths ✅

- [x] `/v1/domains/churn/…` e `/v1/domains/recommendation/…` (predict, runs, promote, deploy, rollback, trigger, sync)
- [x] `ml_core_ring.paths` unificado (`resolve_shared_artifact_path`, `airflow_upload_path`, uploads `ml_data/uploads`)
- [x] `MAP.md` + `docs/DECISOES_REBUILD_DOMINIOS.md`
- [x] Rotas `/processor` mantidas como legado (tag Swagger)
- [ ] Apagar legado — ver checklist em `DECISOES_REBUILD_DOMINIOS.md`

---

- [ ] Side-effect Registry no promote

---

## Fase 7 — Limpeza + frontend-ready

- [ ] Deprecar trilha TC02 CLI-only
- [ ] OpenAPI unificado
- [ ] Remover imports legados violadores

---

## Fase 8 — Buffer / entrega

- [ ] Vídeo STAR, checklist rubrica TC02
