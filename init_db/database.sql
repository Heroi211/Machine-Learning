
CREATE TABLE public.roles (
	id serial4 NOT NULL,
	description varchar NOT NULL,
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
