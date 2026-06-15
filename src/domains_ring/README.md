# domains_ring (anel 2b)

Metadados por **domínio de negócio** — um plugin por pasta.

## Responsabilidade

- `DomainPlugin`: `name`, `problem_type`, `pipeline_runner_id`, schemas, métricas
- Registo em `DOMAIN_REGISTRY` (side-effect no import)

## Pode importar

- `ml_core_ring`, `infra_ring`
- Schemas Pydantic (via `platform_ring/schemas` durante transição)

## Não deve importar

- `executors_ring`, `orchestration_ring`, `platform_ring`

## Código actual (transição)

| Origem | Domínio |
|--------|---------|
| `src/domains/churn/` | churn |
| `src/domains/recommendation/plugin.py` | recommendation |

Implementação pesada de recommendation move para `executors_ring` na Fase 2.
