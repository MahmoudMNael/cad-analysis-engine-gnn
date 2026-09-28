ALTER TABLE jobs ADD COLUMN IF NOT EXISTS notify_email TEXT;

ALTER TABLE jobs DROP CONSTRAINT IF EXISTS jobs_status_check;
ALTER TABLE jobs ADD CONSTRAINT jobs_status_check
  CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled'));
