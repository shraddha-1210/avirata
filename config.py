"""Central tunables for the Avirata pipeline.

Every weight / threshold / window below is an ILLUSTRATIVE demo value. Anything
touching money movement or risk (risk weights, firing threshold, settlement-hold
window) requires actuary / RBI e-mandate + AFA review before production use.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- external credentials / services ---
    google_api_key: str = ""
    tier2_model: str = "gemini-3.5-flash-lite"
    database_url: str = "postgresql+psycopg://avirata:avirata@localhost:5432/avirata"
    # Sized so the Phase 4 concurrent burst gets one real connection per worker
    # rather than queueing at the pool (which would fake a passing result).
    db_pool_size: int = 120
    db_max_overflow: int = 20

    # --- Layer 1: synthetic dataset (fixed seed => reproducible measurement) ---
    dataset_seed: int = 42
    dataset_n: int = 240

    # --- Layer 2: MAD anomaly detection ---
    mad_threshold: float = 3.0          # flag when |value - median| > mad_threshold * dispersion
    min_sample_size: int = 30           # hard gate; below this -> insufficient_data
    # MAD of a count series is legitimately 0 when >50% of days share one count
    # (e.g. [3,3,3,3,4,5,3...]). A 0 threshold would flag every +1 day as an
    # anomaly, so we floor dispersion at one whole decline event — the natural
    # quantum of count data. Illustrative demo value; retune on real traffic.
    mad_dispersion_floor: float = 1.0

    # --- Layer 3: 3-tier diagnosis ---
    tier2_confidence_threshold: float = 0.85   # below -> Tier 3 quarantine
    llm_input_max_chars: int = 200             # raw error string cap before prompt insertion
    tier2_max_tokens: int = 256                # single small JSON object; no room to ramble

    # --- Layer 4: recovery policy + idempotency ---
    risk_weight_urgency: float = 0.2           # illustrative; require actuary review
    risk_weight_reliability: float = 0.3
    # Retuned from 0.4 -> 0.5 (Phase 4 follow-up). Cost-benefit's former 0.1 was
    # folded in here: expected loss was being counted TWICE — once as a score
    # component and again as the hard alt-rail gate. It is now purely a gate,
    # which is also what makes that gate reachable on real events. Weights must
    # still sum to 1.0 (test_weights_sum_to_one).
    risk_weight_amount: float = 0.5
    risk_weight_cost_benefit: float = 0.0
    risk_firing_threshold: float = 0.6         # alt-rail requires score >= this AND cost-benefit pass
    max_retries: int = 2
    ttl_processing_seconds: int = 300          # a case may sit in 'processing' at most this long
    ttl_watchdog_interval_seconds: int = 60    # demo: cron polling; production would be event-driven

    # --- Layer 3 resilience: retries + circuit breaker around the Gemini call ---
    # Retry ONLY what is positively identifiable as transient (network error, 429,
    # 5xx). An unrecognised error class is not retried: re-sending a request whose
    # fate we cannot determine risks duplicating work against a partially-succeeded
    # call, which is worse than one extra quarantine.
    tier2_max_retries: int = 2                 # attempts = 1 + this
    tier2_retry_base_ms: int = 250             # full-jitter base
    tier2_retry_cap_ms: int = 2000             # per-sleep ceiling
    # Circuit breaker. The threshold counts LOGICAL call failures in a rolling
    # window — retries inside one call count once, or a threshold of 3 would really
    # be a threshold of 1. Illustrative demo values; retune on real traffic.
    circuit_failure_threshold: int = 3
    circuit_cooldown_seconds: int = 60
    circuit_half_open_test_calls: int = 1
    circuit_window_seconds: int = 60

    # --- webhook authenticity ---
    # OPT-IN. Unset means every webhook is accepted, which is what keeps the demo
    # runnable with no key material; the app logs a warning at startup so an
    # unsigned deployment is loud rather than silent. Set it in production: these
    # two endpoints move money, and without a shared secret anyone who can reach
    # the port can inject a decline or a settlement.
    webhook_signing_secret: str | None = None

    # --- Layer 5: reconciliation ---
    settlement_hold_seconds: int = 300         # collision window for auto-refund of the losing path


settings = Settings()
