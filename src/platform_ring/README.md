# platform_ring (anel 1)

Contratos de produto implementados em `platform_ring/` (Fase 4).

| Módulo | Valor para o produto |
|--------|----------------------|
| `training_trigger.py` | Um botão de treino para qualquer domínio → Airflow dispatch |
| `promote_service.py` | Promote único por domínio (churn FE ou recomendação) |
| `runs_service.py` | Listagem de runs filtrada por domínio |

Rotas HTTP continuam em `src/api/` (facade); lógica de produto migra gradualmente para aqui.

Ver [PLATFORM_CONTRACT.md](../../docs/PLATFORM_CONTRACT.md).
