"""Outbound job-completion mail. Failures must never fail a job."""

from __future__ import annotations

from email.utils import parseaddr

import emails

from packages.common.configuration import Settings
from packages.common.errors import InvalidRequestError
from packages.common.logging import get_logger

log = get_logger("common.mailer")


def normalize_notify_email(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    _name, addr = parseaddr(cleaned)
    addr = addr.strip()
    if addr != cleaned or "@" not in addr:
        raise InvalidRequestError("invalid email", fields={"email": cleaned})
    local, _, domain = addr.partition("@")
    if not local or not domain or "." not in domain or domain.startswith(".") or domain.endswith("."):
        raise InvalidRequestError("invalid email", fields={"email": cleaned})
    return addr.lower()


def job_result_url(base_url: str, job_id: str) -> str:
    return f"{base_url.rstrip('/')}/jobs/{job_id}"


def notify_analysis_completed(
    settings: Settings,
    *,
    job_id: str,
    notify_email: str | None,
) -> None:
    if not notify_email:
        return
    base = (settings.frontend_trainer_base_url or "").strip()
    host = (settings.smtp_host or "").strip()
    mail_from = (settings.smtp_from or "").strip()
    if not base or not host or not mail_from:
        log.info(
            "notify_skipped",
            extra={"job_id": job_id, "stage": "notify", "error_detail": "mailer not configured"},
        )
        return
    url = job_result_url(base, job_id)
    smtp = {
        "host": host,
        "port": settings.smtp_port,
        "user": settings.smtp_user or None,
        "password": settings.smtp_password or None,
        "tls": bool(settings.smtp_use_tls) and settings.smtp_port != 465,
        "ssl": settings.smtp_port == 465,
    }
    try:
        message = emails.Message(
            text=f"Analysis completed.\n\n{url}",
            subject="Analysis completed",
            mail_from=mail_from,
        )
        response = message.send(to=notify_email, smtp=smtp)
        status = getattr(response, "status_code", None)
        if status not in (None, 250):
            log.error(
                "notify_failed",
                extra={
                    "job_id": job_id,
                    "stage": "notify",
                    "error_detail": f"smtp status={status}",
                },
            )
            return
        log.info("notify_sent", extra={"job_id": job_id, "stage": "notify"})
    except Exception:
        log.exception("notify_failed", extra={"job_id": job_id, "stage": "notify"})
