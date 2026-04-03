# Observabilidade e manutenção

Este documento complementa o [Manual do utilizador](MANUAL_DO_USUARIO.md). Explica **onde** estão os registos (logs), **como** se mede latência no projeto e **quando** entra em jogo a análise de **drift** — com base no código atual.

---

## 1. O que **não** existe na aplicação (importante)

- **Não há** ecrã nem rota de API exclusiva para “o administrador abrir os logs” no browser. Quem opera o sistema precisa de **acesso ao servidor** (ou a cópias dos ficheiros) ou de ferramentas externas (SIEM, agregador de logs) configuradas pela equipa de infraestrutura.
- **Drift** e **relatórios de latência agregada** **não** correm sozinhos dentro do serviço HTTP em cada predição como alertas automáticos. São processos **offline** (scripts) que a equipa pode **agendar** (por exemplo `cron` ou pipeline de CI), conforme descrito em `scripts/maintenance/README.md`.

---

## 2. Onde encontrar os logs

### 2.1 Pedidos HTTP (todas as rotas, incluindo `predict`)

| Item | Detalhe |
|------|---------|
| **Logger** | `api.request` |
| **Ficheiro** | `{PATH_API_REQUEST_LOGS}/access.jsonl` (padrão: `logs/api_requests/access.jsonl`) |
| **Formato** | Uma linha **JSON** por pedido, com `method`, `path`, `status`, `duration_ms` (latência), `client`, `request_id`, `ts`, `error`. |
| **Rotação** | Ficheiros `access.jsonl`, `access.jsonl.1`, … quando o tamanho máximo é atingido (variáveis `LOG_HTTP_REQUESTS_MAX_BYTES`, `LOG_HTTP_REQUESTS_BACK_COUNT`). |
| **Ativar / desativar** | `LOG_HTTP_REQUESTS` (registar no middleware), `LOG_HTTP_REQUESTS_FILE` (escrever em disco). |

O middleware em `core/middleware/request_record.py` mede o tempo **de cada** pedido e grava `duration_ms`. O cabeçalho de resposta `X-Request-ID` correlaciona com o campo `request_id` no JSONL; pode enviar-se `X-Request-ID` no pedido para alinhar com outros sistemas.

### 2.2 Execuções de pipeline (treino baseline / feature engineering via API)

| Item | Detalhe |
|------|---------|
| **Logger** | `ml.pipeline` |
| **Pasta** | `{PATH_DATA}/{PATH_LOGS}/<data_hora_do_run>/` (ex.: `data/logs/20250331_143022/`) |
| **Ficheiro** | `pipeline_<data_hora>.txt` — texto com contexto `run_id`, `objective`, `pipeline_type`. |

Isto é configurado em `setup_pipeline_run_logging` (`core/custom_logger.py`) quando o administrador dispara treinos pelo processador.

### 2.3 Documentação interna adicional

- `docs/LOGS_OPERACAO.md` — resumo dos dois destinos acima e variáveis de ambiente.

---

## 3. Latência: o que é automático vs relatório

| O que acontece | Descrição |
|----------------|-----------|
| **Automático** | Cada pedido HTTP (se `LOG_HTTP_REQUESTS` estiver ativo) regista **latência em milissegundos** (`duration_ms`) no `access.jsonl`. Isto vale para qualquer rota autenticada, incluindo `POST .../processor/predict`. |
| **Relatório agregado** | O script `scripts/maintenance/latency_report.py` **lê** os ficheiros `access.jsonl*` e gera um **CSV** com percentis (mediana, p90, p95, p99, média) em `PATH_MAINTENANCE_REPORTS` (padrão `artifacts/reports/`). **Não** é gerado pela API; é preciso executar o script na máquina onde estão os logs (ou com cópia dos ficheiros). |

Ou seja: a **conferência** de latência é feita por **análise dos ficheiros** ou pelo **CSV** produzido pelo script, não por um painel integrado na aplicação.

---

## 4. Drift: quando acontece e o que significa

| Aspeto | Comportamento atual |
|--------|---------------------|
| **Na API em tempo real** | **Não** há verificação de drift ao chamar `predict`. A predição usa o modelo implantado; não bloqueia nem alerta por drift. |
| **Offline** | O script `scripts/maintenance/drift_report.py` compara um CSV de **treino** (referência) com um CSV de **produção** (em geral exportado da tabela `predictions`, com `input_data` em JSON por linha). Calcula métricas como **PSI** para colunas numéricas alinháveis. |
| **Quando correr** | **Quando a equipa decidir** — por exemplo semanalmente, após deploy, ou quando há suspeita de mudança de dados. O `README` dos scripts sugere agendar com `cron`/CI (“ciclo futuro”). |

Resumo para não técnicos: **drift** é “os dados de hoje ainda parecem os de quando treinámos?” — no projeto isso é uma **análise periódica ou manual**, não um alarme automático em cada predição.

---

## 5. Dados de predição na base (útil para drift e auditoria)

As predições ficam na tabela **`predictions`** (campos como `input_data`, `prediction`, `probability`, `pipeline_run_id`). Para o relatório de drift, costuma exportar-se para CSV (exemplo de SQL no `scripts/maintenance/README.md`).

---

## 6. Resumo para administradores e operações

1. **Logs de tráfego e latência por pedido:** pasta `PATH_API_REQUEST_LOGS` → `access.jsonl`.  
2. **Logs de treino:** pastas sob `PATH_DATA` + `PATH_LOGS` com timestamp.  
3. **Latência agregada:** executar `python scripts/maintenance/latency_report.py` (com logs disponíveis).  
4. **Drift:** exportar predições + ter CSV de treino; executar `drift_report.py`; agendar se fizer sentido para o negócio.  
5. **Evolução futura:** painel na API, alertas automáticos e integração com APM são **fora** do escopo mínimo descrito acima — exigiriam desenvolvimento ou ferramentas externas.
