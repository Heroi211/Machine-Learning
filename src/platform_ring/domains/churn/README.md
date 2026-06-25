# Domínio churn — rotas HTTP

Prefixo Swagger: **`/v1/domains/churn`**

| Rota | Modo |
|------|------|
| `POST /predict` | Manual — body = features Telco |
| `GET /admin/runs` | Manual |
| `POST /admin/promote` | Manual |
| `GET /admin/deployments/history` | Manual |
| `POST /admin/rollback` | Manual |
| `POST /admin/train/trigger` | Auto — Airflow + CSV |
| `POST /admin/train/baseline` | Sync debug (`ENVIRONMENT≠prd`) |
| `POST /admin/train/feature-engineering` | Sync debug |

Código ML: `services/pipelines/`, `executors_ring/tabular_classification/`.
