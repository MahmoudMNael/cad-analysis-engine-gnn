-- P0/P1 schema. Feedback and training tables exist as contracts for later phases.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    source_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    source_uri TEXT NOT NULL,
    unit TEXT,
    unit_source TEXT,
    consent_training BOOLEAN NOT NULL DEFAULT FALSE,
    engine_sha TEXT,
    pipeline_version TEXT,
    model_versions_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    configuration_hash TEXT,
    error_code TEXT,
    error_detail TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    attempt INTEGER NOT NULL DEFAULT 0,
    worker_id TEXT,
    original_filename TEXT,
    snapshot_version INTEGER,
    layout_rev INTEGER NOT NULL DEFAULT 0,
    CONSTRAINT jobs_failed_has_error CHECK (
        status <> 'failed' OR (error_code IS NOT NULL AND error_detail IS NOT NULL)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS jobs_succeeded_snapshot_uidx
    ON jobs (id)
    WHERE status = 'succeeded';

CREATE TABLE IF NOT EXISTS job_artifacts (
    id BIGSERIAL PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    snapshot_version INTEGER NOT NULL DEFAULT 1,
    kind TEXT NOT NULL,
    uri TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (job_id, snapshot_version, kind)
);

CREATE TABLE IF NOT EXISTS feedback_events (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    sequence INTEGER NOT NULL,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    target_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    layout_rev INTEGER NOT NULL,
    retracted_by_event_id TEXT,
    UNIQUE (job_id, sequence)
);

CREATE TABLE IF NOT EXISTS layout_revisions (
    job_id TEXT NOT NULL REFERENCES jobs(id),
    layout_rev INTEGER NOT NULL,
    event_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (job_id, layout_rev)
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    checkpoint_uri TEXT,
    metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL CHECK (
        status IN ('candidate', 'shadow', 'promoted', 'rejected', 'rolled_back')
    ),
    trained_on_engine_sha TEXT,
    feature_schema_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    promoted_at TIMESTAMPTZ,
    PRIMARY KEY (model_name, version)
);

CREATE UNIQUE INDEX IF NOT EXISTS model_registry_one_promoted
    ON model_registry (model_name)
    WHERE status = 'promoted';

CREATE TABLE IF NOT EXISTS training_runs (
    id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    dataset_uri TEXT,
    error_detail TEXT
);
