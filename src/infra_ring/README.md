# infra_ring (anel 0)

Infraestrutura transversal — **sem regra de negócio ML**.

## Responsabilidade

- Settings / `.env` (`configs`)
- SQLAlchemy, sessão async (`database`)
- Logging estruturado
- Clientes externos (MLflow URI, paths partilhados)

## Pode importar

- Bibliotecas standard e third-party

## Não deve importar

- `platform_ring`, `ml_core_ring`, `executors_ring`, `orchestration_ring`

## Migração

| Origem actual | Destino |
|---------------|---------|
| `src/core/configs.py` | `infra_ring/configs.py` |
| `src/core/database.py` | `infra_ring/database.py` |
| `src/core/logging_*.py` | `infra_ring/logging/` |
