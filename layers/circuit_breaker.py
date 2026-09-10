"""A small, dependency-free circuit breaker for the Tier 2 Gemini call.

The failure this exists to make visible: when Gemini is down, rate-limited, or
holding a bad credential, every decline string still "works" — it quarantines. The
pipeline is safe but the dashboard is lying by omission, because a quarantine that
means *the model said it could not map this* looks identical to one that means *we
never asked*. This breaker separates the two, and stops us hammering a dead
dependency once we know it is dead.

Three states, the standard shape:

* **CLOSED** — normal. Calls go through. Failures accumulate in a rolling window;
  reaching `failure_threshold` inside `window_seconds` trips the breaker OPEN.
* **OPEN** — the dependency is presumed down. Calls are skipped without being
  attempted, so the pipeline degrades in milliseconds rather than blocking on a
  timeout per event. After `cooldown_seconds` the breaker moves itself to
  HALF_OPEN.
* **HALF_OPEN** — probation. Exactly `half_open_test_calls` calls are let through
  to find out whether the dependency is back. One success closes the breaker; one
  failure re-opens it and restarts the cooldown.

Two design notes worth stating because they are easy to get wrong:

**A rolling window, not a running total.** Three failures spread over an hour are
not an outage; three inside a minute are. Timestamps outside the window are pruned
before every threshold check, so an occasional blip never accumulates into a trip.

**Monotonic time, not wall clock.** All timing uses `time.monotonic()`, which
cannot jump backwards when NTP corrects the system clock. A wall-clock breaker can
be talked into an infinite OPEN state by a clock skew. Wall-clock timestamps ARE
recorded separately, but only for display — never for a decision.

The clock is injectable (`time_fn`) so tests can drive transitions deterministically
instead of sleeping through a real cooldown.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Literal

logger = logging.getLogger(__name__)

CircuitState = Literal["CLOSED", "OPEN", "HALF_OPEN"]

# (old_state, new_state, snapshot_at_transition) -> None
StateChangeHook = Callable[[CircuitState, CircuitState, "CircuitSnapshot"], None]

CLOSED: CircuitState = "CLOSED"
OPEN: CircuitState = "OPEN"
HALF_OPEN: CircuitState = "HALF_OPEN"


class CircuitOpenError(RuntimeError):
    """Raised by `call()` when the breaker refused to attempt the call.

    Distinct from whatever the wrapped callable raises: this means we never asked,
    which is a different fact about the world than the dependency answering badly.
    """


@dataclass(frozen=True)
class CircuitSnapshot:
    """Point-in-time view for the health endpoint. Safe to serialise."""

    state: CircuitState
    failure_count_in_window: int
    window_seconds: int
    failure_threshold: int
    last_failure_at: str | None
    next_test_at: str | None
    opened_at: str | None
    transitions: int
    half_open_calls_remaining: int
    last_state_change_at: str | None

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "failure_count_last_60s": self.failure_count_in_window,
            "window_seconds": self.window_seconds,
            "failure_threshold": self.failure_threshold,
            "last_failure_at": self.last_failure_at,
            "next_test_at": self.next_test_at,
            "opened_at": self.opened_at,
            "transitions": self.transitions,
            "half_open_calls_remaining": self.half_open_calls_remaining,
            "last_state_change_at": self.last_state_change_at,
        }


class CircuitBreaker:
    """Thread-safe breaker. Every public method takes the lock.

    `allows_call()` mutates state (it is what performs the OPEN -> HALF_OPEN
    transition and consumes a half-open probe slot), so it must be called exactly
    once per attempt and its answer acted on. Use `call()` unless you need to
    interleave other work between the check and the attempt.
    """

    def __init__(
        self,
        *,
        failure_threshold: int,
        cooldown_seconds: float,
        half_open_test_calls: int = 1,
        window_seconds: float = 60.0,
        name: str = "circuit",
        time_fn: Callable[[], float] | None = None,
        on_state_change: StateChangeHook | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if half_open_test_calls < 1:
            raise ValueError("half_open_test_calls must be >= 1")

        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = float(cooldown_seconds)
        self.half_open_test_calls = half_open_test_calls
        self.window_seconds = float(window_seconds)
        self.on_state_change = on_state_change
        # Resolved through the module on every call, not captured at import. A
        # captured reference would hold the ORIGINAL `time.monotonic` and silently
        # ignore any runtime patch of the clock — which is also what makes this
        # testable under freezegun.
        self._now = time_fn if time_fn is not None else (lambda: time.monotonic())

        # RLock, not Lock: `state` and `snapshot` both re-enter the lock via the
        # shared transition helper. A plain Lock would deadlock on itself.
        self._lock = threading.RLock()
        self._failures: deque[float] = deque()
        self._state: CircuitState = CLOSED
        self._opened_at: float | None = None
        self._half_open_remaining = 0
        self._transitions = 0
        self._last_failure_wall: datetime | None = None
        self._opened_at_wall: datetime | None = None
        self._last_state_change_wall: datetime | None = None

    # -- internals (call with the lock held) --------------------------------
    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()

    def _transition(self, new_state: CircuitState, now: float) -> None:
        # The early return is what makes `on_state_change` fire on real
        # transitions only: re-entering the state you are already in is not an
        # event, and alerting on it would train operators to ignore the alert.
        if new_state == self._state:
            return
        old = self._state
        self._state = new_state
        self._transitions += 1
        self._last_state_change_wall = datetime.now(timezone.utc)

        if new_state == OPEN:
            self._opened_at = now
            self._opened_at_wall = datetime.now(timezone.utc)
            self._half_open_remaining = 0
        elif new_state == HALF_OPEN:
            self._half_open_remaining = self.half_open_test_calls
        elif new_state == CLOSED:
            self._opened_at = None
            self._opened_at_wall = None
            self._half_open_remaining = 0
            self._failures.clear()

        # DEBUG, not WARNING: `on_state_change` owns the operator-facing record and
        # its severity split. Logging every transition at WARNING here too would
        # both duplicate that line and stamp a recovery as a warning.
        logger.debug("circuit %r: %s -> %s", self.name, old, new_state)
        self._notify(old, new_state)

    def _notify(self, old: CircuitState, new: CircuitState) -> None:
        """Hand the transition to the observer. Never let it break the caller.

        Invoked with the lock held, so the snapshot the observer receives is the
        one that belongs to this transition rather than a later state. The lock is
        an RLock, so a handler may call back into the breaker; a handler that
        blocks, however, blocks every other thread, so handlers must stay cheap.
        A raising handler is logged and swallowed: alerting is observability, and
        observability must never be able to fail a money-path call.
        """
        if self.on_state_change is None:
            return
        try:
            self.on_state_change(old, new, self._snapshot_locked())
        except Exception:  # noqa: BLE001 - a bad observer must not break the breaker
            logger.exception(
                "circuit %r: on_state_change handler raised on %s -> %s",
                self.name, old, new,
            )

    def _maybe_open_to_half(self, now: float) -> None:
        """OPEN outlives its cooldown -> HALF_OPEN. Lazy, so no timer thread."""
        if self._state == OPEN and self._opened_at is not None:
            if now - self._opened_at >= self.cooldown_seconds:
                self._transition(HALF_OPEN, now)

    # -- public API ---------------------------------------------------------
    @property
    def state(self) -> CircuitState:
        """Current state, after applying any due cooldown transition."""
        with self._lock:
            now = self._now()
            self._maybe_open_to_half(now)
            return self._state

    def allows_call(self) -> bool:
        """May a call be attempted right now? Consumes a half-open probe slot."""
        with self._lock:
            now = self._now()
            self._maybe_open_to_half(now)

            if self._state == CLOSED:
                return True
            if self._state == OPEN:
                return False
            # HALF_OPEN: hand out a bounded number of probes, then hold the rest
            # back. Without this cap every waiting request stampedes a dependency
            # that has just shown one sign of life.
            if self._half_open_remaining > 0:
                self._half_open_remaining -= 1
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            now = self._now()
            if self._state == HALF_OPEN:
                self._transition(CLOSED, now)
            elif self._state == CLOSED:
                # A success is evidence the dependency is healthy; let it erode the
                # window so isolated blips never accumulate into a trip.
                self._failures.clear()

    def record_failure(self) -> None:
        """One LOGICAL call failed. Retries inside that call count once, not once each."""
        with self._lock:
            now = self._now()
            self._last_failure_wall = datetime.now(timezone.utc)

            if self._state == HALF_OPEN:
                self._transition(OPEN, now)
                return

            self._failures.append(now)
            self._prune(now)
            if self._state == CLOSED and len(self._failures) >= self.failure_threshold:
                self._transition(OPEN, now)

    def call(self, fn: Callable[[], object]) -> object:
        """Run `fn` under the breaker. Raises `CircuitOpenError` if it is OPEN."""
        if not self.allows_call():
            raise CircuitOpenError(
                f"circuit {self.name!r} is {self.state} — call not attempted"
            )
        try:
            result = fn()
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result

    def snapshot(self) -> CircuitSnapshot:
        with self._lock:
            now = self._now()
            self._maybe_open_to_half(now)
            self._prune(now)
            return self._snapshot_locked()

    def _snapshot_locked(self) -> CircuitSnapshot:
        """Build a snapshot from current fields. The lock must already be held.

        Split out so `_notify` can hand an observer the snapshot belonging to the
        transition that just happened, without re-entering the transition logic and
        without racing another thread between the change and the read.
        """
        next_test_at = None
        if self._state == OPEN and self._opened_at_wall is not None:
            next_test = self._opened_at_wall.timestamp() + self.cooldown_seconds
            next_test_at = datetime.fromtimestamp(next_test, timezone.utc).isoformat()

        return CircuitSnapshot(
            state=self._state,
            failure_count_in_window=len(self._failures),
            window_seconds=int(self.window_seconds),
            failure_threshold=self.failure_threshold,
            last_failure_at=(
                self._last_failure_wall.isoformat() if self._last_failure_wall else None
            ),
            next_test_at=next_test_at,
            opened_at=(self._opened_at_wall.isoformat() if self._opened_at_wall else None),
            transitions=self._transitions,
            half_open_calls_remaining=self._half_open_remaining,
            last_state_change_at=(
                self._last_state_change_wall.isoformat()
                if self._last_state_change_wall
                else None
            ),
        )

    def reset(self) -> None:
        """Back to a clean CLOSED breaker. For tests and for an operator override."""
        with self._lock:
            self._failures.clear()
            self._state = CLOSED
            self._opened_at = None
            self._opened_at_wall = None
            self._last_failure_wall = None
            self._half_open_remaining = 0
            self._transitions = 0
            self._last_state_change_wall = None
