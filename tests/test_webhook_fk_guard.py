"""Ingest must tolerate a webhook naming a mandate we have never seen.

`decline_events.mandate_id` is a foreign key. A decline for an unknown mandate is a
normal thing for a real webhook to deliver (a mandate created on the PSP side that
we have not backfilled yet), and it must not become a 500 or an IntegrityError.

The concurrency case is the one worth testing properly: two webhooks for the same
unknown mandate arriving together would both pass a read-then-write existence check
and one would still fail on the unique constraint. The guard therefore has to be
`INSERT ... ON CONFLICT DO NOTHING`, and this file asserts that under a real race.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import app as app_module
import store
from db import get_session
from models import DeclineEvent, Mandate

NEW_MANDATE = "MND-FIRST-SEEN-0001"
KNOWN_MANDATE = "MND-ALREADY-KNOWN-1"


@pytest.fixture
def client():
    return TestClient(app_module.app)


def _event(event_id: str, mandate_id: str, **over) -> dict:
    body = {
        "event_id": event_id,
        "mandate_id": mandate_id,
        "customer_id": "CUST-FK-0001",
        "bank": "ICICI",
        "mandate_type": "UPI_AUTOPAY",
        "event_ts": "2026-09-08T10:00:00+00:00",
        "billing_cycle": "2026-09",
        "amount": 999,
        "raw_error_code": "U14",
    }
    body.update(over)
    return body


def test_unknown_mandate_is_auto_created_not_rejected(client, pg_session):
    """The headline: a first-seen mandate ingests cleanly instead of an FK error."""
    assert pg_session.get(Mandate, NEW_MANDATE) is None

    res = client.post("/api/events/ingest", json=_event("EVT-FK-1", NEW_MANDATE))

    assert res.status_code == 200, res.text
    mandate = pg_session.get(Mandate, NEW_MANDATE)
    assert mandate is not None, "the parent row must have been created"
    assert mandate.bank == "ICICI"
    assert mandate.mandate_type == "UPI_AUTOPAY"
    assert mandate.customer_id == "CUST-FK-0001"
    assert mandate.created_at is not None


def test_auto_created_mandate_defaults_to_mid_reliability(client, pg_session):
    """No history means no track record; guessing high would flatter the risk gate."""
    client.post("/api/events/ingest", json=_event("EVT-FK-2", NEW_MANDATE))
    mandate = pg_session.get(Mandate, NEW_MANDATE)
    assert mandate.reliability_score == app_module.AUTOCREATE_RELIABILITY == 0.5


def test_supplied_reliability_is_respected_over_the_default(client, pg_session):
    client.post(
        "/api/events/ingest",
        json=_event("EVT-FK-3", NEW_MANDATE, mandate_reliability=0.82),
    )
    assert pg_session.get(Mandate, NEW_MANDATE).reliability_score == pytest.approx(0.82)


def test_auto_creation_is_logged_as_a_warning(client, pg_session, caplog):
    """An auto-created mandate is a fact an operator should be able to grep for."""
    with caplog.at_level("WARNING"):
        client.post("/api/events/ingest", json=_event("EVT-FK-4", NEW_MANDATE))

    assert any(
        f"auto-created mandate {NEW_MANDATE} on first-seen event EVT-FK-4" in r.getMessage()
        for r in caplog.records
    ), f"expected the auto-create warning, got {[r.getMessage() for r in caplog.records]}"


def test_known_mandate_path_is_unchanged(client, pg_session, caplog):
    """An existing mandate must not be re-created, re-logged, or have its fields reset."""
    store.upsert_mandate(
        pg_session,
        mandate_id=KNOWN_MANDATE,
        customer_id="CUST-ORIGINAL",
        bank="HDFC",
        mandate_type="UPI_AUTOPAY",
        reliability_score=0.97,
    )
    pg_session.commit()

    with caplog.at_level("WARNING"):
        res = client.post(
            "/api/events/ingest",
            json=_event("EVT-FK-5", KNOWN_MANDATE, bank="ICICI", customer_id="CUST-DIFFERENT"),
        )

    assert res.status_code == 200
    pg_session.expire_all()
    mandate = pg_session.get(Mandate, KNOWN_MANDATE)
    assert mandate.customer_id == "CUST-ORIGINAL", "an existing row must not be overwritten"
    assert mandate.bank == "HDFC"
    assert mandate.reliability_score == pytest.approx(0.97)
    assert not any("auto-created mandate" in r.getMessage() for r in caplog.records)


def test_upsert_reports_creation_only_the_first_time(pg_session):
    kw = dict(
        mandate_id=NEW_MANDATE,
        customer_id="CUST-FK-0001",
        bank="ICICI",
        mandate_type="UPI_AUTOPAY",
        reliability_score=0.5,
    )
    assert store.upsert_mandate(pg_session, **kw) is True
    pg_session.commit()
    assert store.upsert_mandate(pg_session, **kw) is False


def test_the_decline_event_actually_lands(client, pg_session):
    """Auto-creating the parent is pointless if the child insert still fails."""
    client.post("/api/events/ingest", json=_event("EVT-FK-6", NEW_MANDATE))
    row = pg_session.execute(
        select(DeclineEvent).where(DeclineEvent.event_id == "EVT-FK-6")
    ).scalar_one_or_none()
    assert row is not None
    assert row.mandate_id == NEW_MANDATE


# ---------------------------------------------------------------------------
# the race
# ---------------------------------------------------------------------------
def test_concurrent_unknown_mandate_webhooks_produce_exactly_one_row(pg_session):
    """Two webhooks for the same unknown mandate: one creates, neither errors.

    A start barrier releases both workers only once each has its own connection,
    so what they race on is the INSERT itself rather than connection setup.
    """
    n = 8
    barrier = threading.Barrier(n)
    errors: list[Exception] = []

    def fire() -> bool:
        session = get_session()
        try:
            session.connection()
            barrier.wait(timeout=30)
            created = store.upsert_mandate(
                session,
                mandate_id=NEW_MANDATE,
                customer_id="CUST-FK-0001",
                bank="ICICI",
                mandate_type="UPI_AUTOPAY",
                reliability_score=0.5,
            )
            session.commit()
            return created
        except Exception as exc:  # noqa: BLE001 - collected and asserted on below
            errors.append(exc)
            session.rollback()
            return False
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(lambda _: fire(), range(n)))

    assert errors == [], f"no worker may fail: {errors}"
    assert sum(results) == 1, f"exactly one insert must win, got {sum(results)}"

    pg_session.expire_all()
    rows = pg_session.execute(
        select(Mandate).where(Mandate.mandate_id == NEW_MANDATE)
    ).scalars().all()
    assert len(rows) == 1
