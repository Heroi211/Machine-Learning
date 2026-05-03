-- Executar em bases já existentes (fora do primeiro init):
--   psql -U ... -d ... -f init_db/add_pipeline_runs_is_airflow_run.sql

ALTER TABLE public.pipeline_runs
ADD COLUMN IF NOT EXISTS is_airflow_run boolean NOT NULL DEFAULT false;

COMMENT ON COLUMN public.pipeline_runs.is_airflow_run IS 'true = criado pelo DAG Airflow; false = rotas síncronas / admin.';
