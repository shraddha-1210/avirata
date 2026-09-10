"""Database access for the ingest -> detection path.

Kept separate from `app.py` so route logic can be tested with this module
patched, and so every raw SQL/ORM write lives in one auditable place.

Idempotency note: webhooks retry. `insert_decline_event` and `upsert_mandate`
both use `INSERT ... ON CONFLICT DO NOTHING` against real UNIQUE/PK
constraints — never an application-level lock.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from layers.detection import AnomalyResult
from models import DeclineEvent, DetectedAnomaly, Diagnosis, Mandate, QuarantineQueue


def upsert_mandate(
    session: Session,
    *,
    mandate_id: str,
    customer_id: str,
    bank: str,
    mandate_type: str,
    reliability_score: float,
) -> bool:
    """Ensure a mandate row exists. Returns True if THIS call created it.

    `decline_events.mandate_id` is a foreign key, so a webhook naming a mandate we
    have never seen would otherwise fail the insert. Creating the parent row here
    means a first-seen mandate is a normal event rather than a 500.

    The return value is what lets the caller log the auto-creation. It comes from
    `RETURNING` on the INSERT rather than a preceding SELECT: two concurrent
    webhooks for the same unknown mandate would both pass a read-then-write check
    and one would still raise. With `ON CONFLICT DO NOTHING ... RETURNING`, the
    loser gets no row back and reports False, and neither call fails.
    """
    stmt = (
        pg_insert(Mandate)
        .values(
            mandate_id=mandate_id,
            customer_id=customer_id,
            bank=bank,
            mandate_type=mandate_type,
            reliability_score=reliability_score,
        )
        .on_conflict_do_nothing(index_elements=["mandate_id"])
        .returning(Mandate.mandate_id)
    )
    return session.execute(stmt).scalar_one_or_none() is not None


def insert_decline_event(session: Session, *, event: dict) -> bool:
    """Insert one decline event. Returns False if it was already ingested."""
    stmt = (
        pg_insert(DeclineEvent)
        .values(**event)
        .on_conflict_do_nothing(index_elements=["event_id"])
        .returning(DeclineEvent.event_id)
    )
    return session.execute(stmt).scalar_one_or_none() is not None


def segment_daily_history(
    session: Session, *, segment: str, as_of: date, lookback_days: int = 60
) -> tuple[list[float], float]:
    """Return (prior daily decline counts, count on `as_of`) for one segment.

    Days with no declines are materialised as real zeros across the lookback
    span — a quiet day is an observation, not missing data. Dropping them would
    inflate the baseline median and suppress genuine spikes.
    """
    window_start = as_of - timedelta(days=lookback_days)
    day = func.date(DeclineEvent.event_ts).label("day")
    rows = session.execute(
        select(day, func.count().label("n"))
        .where(DeclineEvent.segment == segment)
        .where(func.date(DeclineEvent.event_ts) >= window_start)
        .where(func.date(DeclineEvent.event_ts) <= as_of)
        .group_by(day)
    ).all()

    counts: dict[date, int] = {r.day: int(r.n) for r in rows}
    history = [
        float(counts.get(window_start + timedelta(days=i), 0))
        for i in range((as_of - window_start).days)
    ]
    observed = float(counts.get(as_of, 0))
    return history, observed


def insert_detected_anomaly(
    session: Session,
    *,
    segment: str,
    window_start: datetime,
    window_end: datetime,
    result: AnomalyResult,
) -> int:
    row = DetectedAnomaly(
        segment=segment,
        window_start=window_start,
        window_end=window_end,
        sample_size=result.sample_size,
        median=result.median,
        mad=result.mad,
        threshold=result.threshold,
        observed_value=result.observed_value,
        is_anomaly=result.is_anomaly,
        status=result.status,
    )
    session.add(row)
    session.flush()
    return row.id


# ---------------------------------------------------------------------------
# Phase 4 — diagnosis + quarantine persistence.
# ---------------------------------------------------------------------------
def insert_diagnosis(session: Session, *, event_id: str, result) -> int:
    """Persist one diagnosis with its full tier/confidence trail."""
    row = Diagnosis(
        event_id=event_id,
        tier=result.tier,
        cause=result.cause,
        confidence=result.confidence,
        status=result.status,
        raw_input=result.raw_input,
        sanitized_input=result.sanitized_input,
        llm_model=result.llm_model,
    )
    session.add(row)
    session.flush()
    return row.id


def insert_quarantine(session: Session, *, event_id: str, result) -> int:
    """Queue an undiagnosed event for Ops review / future ontology promotion."""
    row = QuarantineQueue(
        event_id=event_id,
        raw_input=result.raw_input,
        tier_attempted=result.tier,
        reason=result.reason[:64],
        status="pending_ops_review",
    )
    session.add(row)
    session.flush()
    return row.id
