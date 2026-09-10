"""HMAC-SHA256 authenticity on the two money-touching webhook endpoints.

Verification is opt-in: with no secret configured every request is accepted, which
is what keeps the demo runnable without key material. That default is the risky
one, so it is tested explicitly rather than assumed — including the startup warning
that makes an unsigned deployment loud instead of silent.

The signature covers the RAW body bytes. Signing a re-serialised model instead
would ignore anything the parse discards, and an attacker can vary exactly that.
"""
from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import WEBHOOK_SIGNATURE_HEADER
from config import settings

SECRET = "testsecret"

INGEST_BODY = {
    "event_id": "EVT-SIG-1",
    "mandate_id": "MND-SIG-0001",
    "customer_id": "CUST-SIG",
    "bank": "ICICI",
    "mandate_type": "UPI_AUTOPAY",
    "event_ts": "2026-09-08T10:00:00+00:00",
    "billing_cycle": "2026-09",
    "amount": 999,
    "raw_error_code": "U14",
}

SETTLEMENT_BODY = {
    "mandate_id": "MND-SIG-0001",
    "billing_cycle": "2026-09",
    "path": "mandate",
    "amount": 999,
}


@pytest.fixture
def client():
    return TestClient(app_module.app)


@pytest.fixture
def signing_enabled(monkeypatch):
    monkeypatch.setattr(settings, "webhook_signing_secret", SECRET)
    return SECRET


@pytest.fixture
def signing_disabled(monkeypatch):
    monkeypatch.setattr(settings, "webhook_signing_secret", None)


def sign(body: dict, secret: str = SECRET) -> tuple[bytes, str]:
    """Return the exact bytes to send and their signature. Order matters."""
    raw = json.dumps(body).encode("utf-8")
    return raw, hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()


def _post(client: TestClient, path: str, raw: bytes, signature: str | None):
    headers = {"Content-Type": "application/json"}
    if signature is not None:
        headers[WEBHOOK_SIGNATURE_HEADER] = signature
    return client.post(path, content=raw, headers=headers)


# ---------------------------------------------------------------------------
# enabled
# ---------------------------------------------------------------------------
def test_missing_header_is_rejected(client, signing_enabled, pg_session):
    raw, _ = sign(INGEST_BODY)
    res = _post(client, "/api/events/ingest", raw, None)
    assert res.status_code == 401
    assert WEBHOOK_SIGNATURE_HEADER in res.json()["detail"]


def test_wrong_signature_is_rejected(client, signing_enabled, pg_session):
    raw, _ = sign(INGEST_BODY)
    res = _post(client, "/api/events/ingest", raw, "deadbeef" * 8)
    assert res.status_code == 401
    assert "invalid webhook signature" in res.json()["detail"]


def test_signature_from_the_wrong_secret_is_rejected(client, signing_enabled, pg_session):
    raw, sig = sign(INGEST_BODY, secret="not-the-real-secret")
    assert _post(client, "/api/events/ingest", raw, sig).status_code == 401


def test_correct_signature_is_accepted(client, signing_enabled, pg_session):
    raw, sig = sign(INGEST_BODY)
    res = _post(client, "/api/events/ingest", raw, sig)
    assert res.status_code == 200, res.text
    assert res.json()["event_id"] == "EVT-SIG-1"


def test_tampered_body_invalidates_a_valid_signature(client, signing_enabled, pg_session):
    """The signature must cover the bytes, not the parsed model."""
    raw, sig = sign(INGEST_BODY)
    tampered = dict(INGEST_BODY, amount=999_999)
    res = _post(client, "/api/events/ingest", json.dumps(tampered).encode(), sig)
    assert res.status_code == 401


def test_settlement_webhook_is_also_protected(client, signing_enabled, pg_session):
    raw, _ = sign(SETTLEMENT_BODY)
    assert _post(client, "/api/webhooks/settlement", raw, None).status_code == 401

    raw, sig = sign(SETTLEMENT_BODY)
    assert _post(client, "/api/webhooks/settlement", raw, sig).status_code == 200


# ---------------------------------------------------------------------------
# scope: only the money-touching routes
# ---------------------------------------------------------------------------
def test_ui_and_readonly_routes_stay_open(client, signing_enabled, pg_session):
    """Signing these would mean shipping the secret to the browser."""
    for path in (
        "/api/health",
        "/api/health/gemini",
        "/api/dashboard/summary",
        "/api/ontology/rules",
        "/api/ontology/promoted-rules",
    ):
        assert client.get(path).status_code == 200, f"{path} must not require a signature"


def test_recover_route_is_not_signature_protected(client, signing_enabled, pg_session):
    """Internal, called by the UI right after ingest; a 401 here breaks the console.

    The event is ingested first (signed, since verification is on for this test)
    because `/api/events/recover` writes a diagnosis row that references it.
    """
    raw, sig = sign(INGEST_BODY)
    assert _post(client, "/api/events/ingest", raw, sig).status_code == 200

    res = client.post(
        "/api/events/recover",
        json={
            "event_id": INGEST_BODY["event_id"],
            "mandate_id": INGEST_BODY["mandate_id"],
            "billing_cycle": INGEST_BODY["billing_cycle"],
            "raw_error_code": "U14",
            "amount": 100,
            "mandate_reliability": 0.5,
            "days_to_next_cycle": 3,
        },
    )
    assert res.status_code == 200, "recover must not require a webhook signature"


# ---------------------------------------------------------------------------
# disabled (the demo default)
# ---------------------------------------------------------------------------
def test_unsigned_requests_are_accepted_when_no_secret_is_set(
    client, signing_disabled, pg_session
):
    """Backward compatibility: the demo must keep working with no key material."""
    raw, _ = sign(INGEST_BODY)
    assert _post(client, "/api/events/ingest", raw, None).status_code == 200


def test_a_bogus_signature_is_ignored_when_verification_is_off(
    client, signing_disabled, pg_session
):
    raw, _ = sign(INGEST_BODY)
    assert _post(client, "/api/events/ingest", raw, "not-a-signature").status_code == 200


def test_startup_warns_once_when_verification_is_disabled(signing_disabled, caplog):
    """An unsigned deployment has to be loud, or nobody will notice it is unsigned."""
    with caplog.at_level("WARNING"):
        with TestClient(app_module.app):
            pass

    hits = [
        r for r in caplog.records
        if "webhook signature verification disabled" in r.getMessage()
    ]
    assert len(hits) == 1, f"expected exactly one startup warning, got {len(hits)}"


def test_startup_is_quiet_about_signing_when_a_secret_is_set(signing_enabled, caplog):
    with caplog.at_level("WARNING"):
        with TestClient(app_module.app):
            pass

    assert not any(
        "webhook signature verification disabled" in r.getMessage() for r in caplog.records
    )
