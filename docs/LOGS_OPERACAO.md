# Operação de logs — Atlas-Pipeline

## Dois destinos principais

| Destino | Logger | Arquivo típico | Quando |
|---------|--------|------------------|--------|
| **Rotas HTTP** | `api.request` | `PATH_API_REQUEST_LOGS/access.jsonl` (rotação) | Cada requisição (se `LOG_HTTP_REQUESTS` / `LOG_HTTP_REQUESTS_FILE` ativos) |
| **Pipelines ML** | `ml.pipeline` | `PATH_DATA/PATH_LOGS/<timestamp>/pipeline_<timestamp>.txt` | Cada execução de baseline ou feature engineering (incluindo via API) |

## Formato

### `access.jsonl` (uma linha JSON por requisição)

Campos estáveis: `ts`, `request_id`, `method`, `path`, `status`, `client`, `duration_ms`, `error`.

- Envie header opcional `X-Request-ID` para correlacionar com outros sistemas; se omitido, o servidor gera um UUID e devolve o mesmo valor no header de resposta `X-Request-ID`.

### `pipeline_*.txt` (texto)

Cada linha inclui: `run_id`, `objective`, `pipeline_type` e a mensagem do pipeline, além de timestamp e nível.

## Scripts standalone (`python -m ...`)

`setup_log` em `core/custom_logger` ainda pode reconfigurar o **logger raiz** (console + arquivo) para execução isolada. **Não** chame `setup_log` dentro do worker Uvicorn; use o fluxo da API que chama `setup_pipeline_run_logging`.

## Variáveis de ambiente relevantes

- `PATH_DATA`, `PATH_LOGS`, `PATH_API_REQUEST_LOGS`
- `LOG_HTTP_REQUESTS`, `LOG_HTTP_REQUESTS_FILE`, `LOG_HTTP_REQUESTS_MAX_BYTES`, `LOG_HTTP_REQUESTS_BACKUP_COUNT`
