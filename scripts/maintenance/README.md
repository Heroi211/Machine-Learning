# Scripts de manutenção (offline)

Executar a partir da raiz do repositório (ou com `PYTHONPATH=.`).

## Variáveis úteis (`.env`)

- `PATH_API_REQUEST_LOGS` — diretório do `access.jsonl`
- `PATH_MAINTENANCE_REPORTS` — onde gravar CSV/resumos (padrão `artifacts/reports`)
- Para drift: `PATH_DATA` ou caminho explícito ao CSV de treino

## Latência

```bash
python scripts/maintenance/latency_report.py
```

Lê `access.jsonl` e rotacionados `access.jsonl.*`, agrega `duration_ms` e grava percentis em `artifacts/reports`.

## Drift (treino vs produção)

1. Exportar predições do PostgreSQL, por exemplo:

```sql
COPY (SELECT id, pipeline_run_id, input_data, prediction, probability FROM predictions) TO STDOUT WITH CSV HEADER;
```

Ou guardar CSV com coluna `input_data` (JSON por linha) ou features já “achatadas”.

2. Executar:

```bash
python scripts/maintenance/drift_report.py --train-csv data/heart.csv --predictions-csv exports/predictions.csv
```

O script tenta expandir `input_data` se existir; compara colunas numéricas comuns com o treino e calcula PSI.

## Ciclo futuro

Agendar com `cron`/CI após deploy ou semanalmente; opcionalmente anexar pasta `artifacts/reports/` ao artefato da entrega.
