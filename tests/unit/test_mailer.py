from __future__ import annotations

from types import SimpleNamespace

from packages.common.configuration import Settings
from packages.common.errors import InvalidRequestError
from packages.common.mailer import job_result_url, normalize_notify_email, notify_analysis_completed


def test_normalize_email_blank_is_omitted():
    assert normalize_notify_email(None) is None
    assert normalize_notify_email("") is None
    assert normalize_notify_email("   ") is None


def test_normalize_email_accepts_and_lowercases():
    assert normalize_notify_email("User@Example.COM") == "user@example.com"


def test_normalize_email_rejects_invalid():
    for value in ("not-an-email", "foo@", "@bar.com", "foo@bar", "a b@c.com"):
        try:
            normalize_notify_email(value)
            raise AssertionError(f"expected invalid: {value}")
        except InvalidRequestError as exc:
            assert "email" in exc.fields


def test_job_result_url_strips_trailing_slash():
    assert (
        job_result_url("https://trainer.example/", "abc-uuid")
        == "https://trainer.example/jobs/abc-uuid"
    )


def test_notify_skipped_when_unconfigured():
    settings = Settings(
        frontend_trainer_base_url="",
        smtp_host="",
        smtp_from="",
    )
    notify_analysis_completed(settings, job_id="job-1", notify_email="user@example.com")


def test_notify_skipped_without_recipient():
    settings = Settings(
        frontend_trainer_base_url="https://trainer.example",
        smtp_host="smtp.example.com",
        smtp_from="noreply@example.com",
    )
    notify_analysis_completed(settings, job_id="job-1", notify_email=None)


def test_notify_sends_message(monkeypatch):
    sent: dict = {}

    class FakeMessage:
        def __init__(self, **kwargs):
            sent["init"] = kwargs

        def send(self, to, smtp):
            sent["to"] = to
            sent["smtp"] = smtp
            return SimpleNamespace(status_code=250)

    monkeypatch.setattr("packages.common.mailer.emails.Message", FakeMessage)
    settings = Settings(
        frontend_trainer_base_url="https://trainer.example/",
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_user="mailer",
        smtp_password="secret",
        smtp_from="noreply@example.com",
        smtp_use_tls=True,
    )
    notify_analysis_completed(settings, job_id="job-uuid", notify_email="user@example.com")
    assert sent["to"] == "user@example.com"
    assert "Analysis completed" in sent["init"]["text"]
    assert "https://trainer.example/jobs/job-uuid" in sent["init"]["text"]
    assert sent["init"]["subject"] == "Analysis completed"
    assert sent["smtp"]["host"] == "smtp.example.com"
    assert sent["smtp"]["tls"] is True
    assert sent["smtp"]["ssl"] is False


def test_notify_swallows_smtp_errors(monkeypatch):
    class Boom:
        def __init__(self, **kwargs):
            pass

        def send(self, to, smtp):
            raise RuntimeError("smtp down")

    monkeypatch.setattr("packages.common.mailer.emails.Message", Boom)
    settings = Settings(
        frontend_trainer_base_url="https://trainer.example",
        smtp_host="smtp.example.com",
        smtp_from="noreply@example.com",
    )
    notify_analysis_completed(settings, job_id="job-1", notify_email="user@example.com")
