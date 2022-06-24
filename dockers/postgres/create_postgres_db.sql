CREATE USER admin WITH PASSWORD 'admin';
CREATE DATABASE thesis WITH OWNER=admin
                                  LC_COLLATE='en_US.utf8'
                                  LC_CTYPE='en_US.utf8'
                                  ENCODING='UTF8'
                                  TEMPLATE=template0;
GRANT ALL ON ALL TABLES IN SCHEMA public TO admin;
alter role admin createdb;
GRANT ALL PRIVILEGES ON DATABASE thesis TO admin;

\connect thesis

CREATE TYPE public.lossfunction AS ENUM (
    'mse',
    'mae',
    'huber'
);


ALTER TYPE public.lossfunction OWNER TO admin;

CREATE TABLE public.training_metrics (
    id integer NOT NULL,
    layers integer,
    neurons integer,
    timesteps integer,
    dropout double precision,
    mse double precision,
    mae double precision,
    huber double precision,
    loss_function public.lossfunction,
    mape double precision,
    mse_std double precision,
    mae_std double precision,
    huber_std double precision,
    mape_std double precision,
    created_at timestamp with time zone DEFAULT now(),
);


ALTER TABLE public.training_metrics OWNER TO admin;

--
-- Name: training_metrics_id_seq; Type: SEQUENCE; Schema: public; Owner: admin
--

CREATE SEQUENCE public.training_metrics_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.training_metrics_id_seq OWNER TO admin;

--
-- Name: training_metrics_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: admin
--

ALTER SEQUENCE public.training_metrics_id_seq OWNED BY public.training_metrics.id;
ALTER TABLE ONLY public.training_metrics ALTER COLUMN id SET DEFAULT nextval('public.training_metrics_id_seq'::regclass);

--
-- Name: training_metrics_id_seq; Type: SEQUENCE SET; Schema: public; Owner: admin
--

SELECT pg_catalog.setval('public.training_metrics_id_seq', 1, false);

--
-- Name: training_metrics training_metrics_pkey; Type: CONSTRAINT; Schema: public; Owner: admin
--

ALTER TABLE ONLY public.training_metrics
    ADD CONSTRAINT training_metrics_pkey PRIMARY KEY (id);

--
-- Name: ix_training_metrics_id; Type: INDEX; Schema: public; Owner: admin
--

CREATE INDEX ix_training_metrics_id ON public.training_metrics USING btree (id);

