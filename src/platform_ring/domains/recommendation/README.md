# Domínio recommendation — rotas HTTP

Prefixo Swagger: **`/v1/domains/recommendation`**

| Rota | Modo |
|------|------|
| `POST /predict` | Manual — body `{ user_id, top_k }` |
| `GET /admin/runs` | Manual |
| `POST /admin/promote` | Manual |
| `GET /admin/deployments/history` | Manual |
| `POST /admin/rollback` | Manual |
| `POST /admin/train/trigger` | Auto — Airflow → worker |
| `POST /admin/train/sync` | Sync debug — worker ou in-process |

Código ML: `domains/recommendation/`, worker `:8010`.
