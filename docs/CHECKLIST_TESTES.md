# Checklist de testes do sistema

Use este ficheiro para validar o ambiente após `docker compose up`.  
**Base da API:** `http://localhost:8000` + valor de `PROJECT_VERSION` no `.env` (ex.: `/v1` → prefixo `http://localhost:8000/v1`).

---

## Fase A — Ambiente

- [ ] **A1** — `docker compose ps -a`: serviços de longa duração `Up`; `airflow_init` pode estar `Exited (0)`.
- [ ] **A2** — API: `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/docs` → `200`.
- [ ] **A3** — Airflow: `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/health` → `200`.
- [ ] **A4** — Browser: [pgAdmin](http://localhost:5050) e [Dozzle](http://localhost:8888) abrem.
- [ ] **A5** — Postgres (sem cliente no host):  
  `docker compose exec db_processing sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1;"'`
- [ ] **A6** — Sem erros de import nos DAGs:  
  `docker compose exec airflow-scheduler airflow dags list-import-errors` → saída vazia.

---

## Fase B — Autenticação (Swagger)

Abrir `http://localhost:8000/docs` e usar o prefixo da API (ex.: `/v1`).

- [ ] **B1** — `POST /v1/auth/signup` — registo com JSON (nome, email, password).
- [ ] **B2** — `POST /v1/auth/authenticate` — form OAuth2: `username` = email, `password` — obter `access_token`.
- [ ] **B3** — Authorize com `Bearer <token>` — `GET /v1/auth/logged` → `200` com dados do utilizador.
- [ ] **B4** — `GET /v1/users/` — `200` (lista de utilizadores).

*(Ajusta `/v1` se o teu `PROJECT_VERSION` for outro.)*

---

## Fase C — Perfil administrador

Rotas `/processor/admin/*` exigem `role_id = 2` (Administrador).

- [ ] **C1** — Promover utilizador no Postgres (substituir o email):  
  `docker compose exec db_processing sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "UPDATE users SET role_id = 2 WHERE email = '\''seu@email.com'\'';"'`
- [ ] **C2** — Voltar a autenticar e usar o **novo** token no Swagger.
- [ ] **C3** — Confirmar que chamadas a rotas `.../processor/admin/...` já não devolvem `403`.

---

## Fase D — ML (domínio exemplo: `heart_disease`)

Utilizar CSV compatível (ex.: ficheiros em `data/pre_processed/`; colunas mínimas do domínio, p.ex. `age`, `chol`).

- [ ] **D1** — `POST /v1/processor/admin/train/baseline` — multipart: `file` + `objective=heart_disease` → `201` + CSV; anotar `X-Pipeline-Run-Id` (ou id do run).
- [ ] **D2** — *(Opcional)* `POST /v1/processor/admin/train/feature-engineering` — mesmo padrão + parâmetros de form se necessário.
- [ ] **D3** — `POST /v1/processor/admin/promote` — JSON: `domain`, `pipeline_run_id` → `201`.
- [ ] **D4** — `GET /v1/processor/admin/deployments/heart_disease/history` — lista de deployments.
- [ ] **D5** — `POST /v1/processor/predict` — JSON com `domain` e `features` coerentes com o modelo → `200`. *(Sem promoção ativa, espera-se `404`.)*

---

## Fase E — Airflow

- [ ] **E1** — `POST /v1/processor/admin/train/trigger-dag` — multipart: CSV + `objective` + restantes forms → `202`.
- [ ] **E2** — [UI Airflow](http://localhost:8080) — DAG `ml_training_pipeline` → Grid — run concluído sem falha nas tasks.
- [ ] **E3** — Revisão de logs (Dozzle ou `docker compose logs`) para erros no scheduler/webserver durante o run.

---

## Fase F — Rollback (opcional)

- [ ] **F1** — `POST /v1/processor/admin/rollback` — body `{"domain": "heart_disease"}` → `200`.
- [ ] **F2** — `GET /v1/processor/admin/deployments/heart_disease/history` — estado coerente após rollback.

---

## Critério de conclusão

- Fases **A** e **B** OK.
- **C** com admin ativo e token atualizado.
- **D** com treino → promote → predict com sucesso (fluxo mínimo do ML).
- **E** com DAG a correr sem erro crítico nas tasks.

---

## Referência rápida de URLs

| Serviço   | URL |
|-----------|-----|
| Swagger   | http://localhost:8000/docs |
| Airflow   | http://localhost:8080 |
| pgAdmin   | http://localhost:5050 |
| Dozzle    | http://localhost:8888 |

Credenciais Airflow por defeito no projeto: utilizador/senha definidos no `.env` (`AIRFLOW_USER` / `AIRFLOW_PASSWORD`), frequentemente `airflow` / `airflow`.
