
CREATE TABLE public.roles (
	id serial4 NOT NULL,
	description varchar NOT NULL,
	created_at timestamp NOT NULL DEFAULT NOW(),
	active bool NOT NULL,
	CONSTRAINT roles_pkey PRIMARY KEY (id)
);

CREATE TABLE public.users (
	id serial4 NOT NULL,
	"password" varchar(255) NOT NULL,
	"name" varchar(50) NOT NULL,
	email varchar(50) NOT NULL,
	created_at timestamp NULL,
	role_id int4 NULL,
	active bool NULL,
	CONSTRAINT users_pkey PRIMARY KEY (id)
);

INSERT INTO public.roles (description,active) VALUES
	 ('User',true),
	 ('Administrator',true);

ALTER TABLE public.users ADD CONSTRAINT users_role_id_fkey FOREIGN KEY (role_id) REFERENCES public.roles(id);

INSERT INTO public.users (id, "password", "name", email, created_at, role_id, active)
VALUES
(
    1,
    '$2b$12$.DEdGnH7ht7FavrdBTsmsuP5KQUc7Wez3V.vG7HcTP965efchwuuu',
    'Gabriel Drumond',
    'gabriel.drumond@cod3bit.com.br',
    '2026-05-03 10:55:06.069028',
    2,
    true
),
(
    2,
    '$2b$12$Bv2eIuzxqIS0nvn2ZIXUKepC/cu8xmt9L3i2mM2Bg2bO0AQiygqB.',
    'airflow',
    'airflow@airflow.com.br',
    '2026-05-03 10:56:47.493325',
    2,
    true
);

CREATE TABLE public.pipeline_runs (
	id serial4 NOT NULL,
	user_id int4 NOT NULL,
	pipeline_type varchar(50) NOT NULL,
	is_airflow_run bool NOT NULL DEFAULT false,
	objective varchar(100) NOT NULL,
	status varchar(20) NOT NULL DEFAULT 'processing',
	original_filename varchar(255) NOT NULL,
	model_path varchar(500) NULL,
	csv_output_path varchar(500) NULL,
	metrics jsonb NULL,
	error_message varchar(1000) NULL,
	created_at timestamp NOT NULL DEFAULT NOW(),
	completed_at timestamp NULL,
	active bool NOT NULL DEFAULT true,
	CONSTRAINT pipeline_runs_pkey PRIMARY KEY (id),
	CONSTRAINT pipeline_runs_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id)
);

CREATE TABLE public.predictions (
	id serial4 NOT NULL,
	user_id int4 NOT NULL,
	pipeline_run_id int4 NOT NULL,
	input_data jsonb NOT NULL,
	prediction int4 NOT NULL,
	probability float8 NULL,
	created_at timestamp NOT NULL DEFAULT NOW(),
	active bool NOT NULL DEFAULT true,
	CONSTRAINT predictions_pkey PRIMARY KEY (id),
	CONSTRAINT predictions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id),
	CONSTRAINT predictions_pipeline_run_id_fkey FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id)
);

CREATE TABLE public.deployed_models (
	id serial4 NOT NULL,
	domain varchar(100) NOT NULL,
	pipeline_run_id int4 NOT NULL,
	status varchar(20) NOT NULL DEFAULT 'active',
	promoted_at timestamp NULL,
	promoted_by_user_id int4 NULL,
	metrics_snapshot jsonb NULL,
	created_at timestamp NOT NULL DEFAULT NOW(),
	active bool NOT NULL DEFAULT true,
	CONSTRAINT deployed_models_pkey PRIMARY KEY (id),
	CONSTRAINT deployed_models_pipeline_run_id_fkey FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id),
	CONSTRAINT deployed_models_promoted_by_user_id_fkey FOREIGN KEY (promoted_by_user_id) REFERENCES public.users(id)
);

CREATE INDEX ix_deployed_models_domain ON public.deployed_models (domain);

-- No máximo um deployment ativo por domínio (alinhado a deployment_service.get_active_deployment)
CREATE UNIQUE INDEX uq_deployed_models_one_active_per_domain ON public.deployed_models (domain)
WHERE status = 'active' AND active IS TRUE;
