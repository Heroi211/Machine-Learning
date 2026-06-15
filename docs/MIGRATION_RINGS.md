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

## Fase 1 — `ml_core_ring` + registries

- [ ] Mover `src/core/ml/` → `src/ml_core_ring/` (aliases temporários)
- [ ] `train_backend.py` + `TRAIN_BACKEND_REGISTRY`
- [ ] `orchestration_hooks.py` (Airflow → backend)
- [ ] Testes unitários registries

---

## Fase 2 — `executors_ring`

- [ ] `executors_ring/tabular_classification/` wrap Baseline + FE
- [ ] `executors_ring/recommendation/` wrap runner + worker
- [ ] Registar backends no registry

---

## Fase 3 — `orchestration_ring` (Airflow central)

- [ ] DAG `ml_training_dispatch` com `domain` no conf
- [ ] Tasks tabular + recommendation (DVC/worker)
- [ ] `persist_run` partilhada → `pipeline_runs`

---

## Fase 4 — `platform_ring`

- [ ] Mover API, deployments, runs, schemas
- [ ] `trigger-dag` genérico por `domain`
- [ ] Promote generalizado + `/predict` recommendation

---

## Fase 5 — Worker recommendation

- [ ] Container `worker-recommendation`
- [ ] Compose integrado

---

## Fase 6 — DVC + MLflow Registry no promote

- [ ] Side-effect Registry no promote

---

## Fase 7 — Limpeza + frontend-ready

- [ ] Deprecar trilha TC02 CLI-only
- [ ] OpenAPI unificado
- [ ] Remover imports legados violadores

---

## Fase 8 — Buffer / entrega

- [ ] Vídeo STAR, checklist rubrica TC02
