# platform_ring (anel 1)

**Camada de produto imutável** — contrato [PLATFORM_CONTRACT.md](../../docs/PLATFORM_CONTRACT.md).

## Responsabilidade

- API HTTP (FastAPI)
- Autenticação JWT, users, roles
- `pipeline_runs` (leitura/escrita de metadados)
- Promote, rollback, histórico de deployments
- `/predict` (delega inferência a `ml_core_ring`)
- Schemas OpenAPI públicos (frontend)

## Pode importar

- `infra_ring`
- `ml_core_ring` (apenas registries: `get_domain`, `get_engine`, `ArtifactManifest`)

## Não deve importar

- `executors_ring` — **proibido**
- `orchestration_ring` — disparo via HTTP Airflow, não import de DAGs

## Migração

| Origem actual | Destino |
|---------------|---------|
| `src/api/` | `platform_ring/api/` |
| `src/services/processor/` | `platform_ring/runs/`, `deployments/`, `predictions/` |
| `src/services/auth/` | `platform_ring/auth/` |
| `src/schemas/` | `platform_ring/schemas/` |
