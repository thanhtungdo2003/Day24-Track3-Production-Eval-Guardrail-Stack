# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Student:** Lab submission
**Date:** 2026-08-26

## Guard Stack Architecture

User input flows through local PII scanning, the input rail, the Lab 18 RAG pipeline, and the output rail. PII is anonymized or blocked before retrieval; jailbreak, prompt-injection, and off-topic requests are refused; sensitive output is replaced with a safe response.

## Latency Budget

| Layer | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---:|---:|---:|
| Presidio/regex PII | 0.01 | 0.02 | 0.02 |
| NeMo/input fallback | 0.07 | 0.19 | 0.19 |
| Total guard | 0.08 | 0.22 | 0.22 |

**Budget OK:** Yes; target is < 500 ms P95.

## CI/CD Gates

Run `uv run pytest tests/`, `uv run python src/phase_a_ragas.py`, and `uv run python generate_submission.py`. Merge only when all tests pass, the RAG report contains 50 questions, adversarial pass rate is at least 75%, and guard P95 remains below 500 ms.

## Monitoring Dashboard

Alert when daily faithfulness is below 0.70, adversarial block rate is below 80%, guard P95 exceeds 600 ms, or PII detections spike above 10 per hour. Investigate the corresponding report and review new attack patterns before changing thresholds.

## Results

| Metric | Result |
|---|---:|
| RAGAS avg score | 0.7032 |
| Factual / multi-hop / adversarial average | 0.7930 / 0.6462 / 0.6378 |
| Dominant failure distribution | factual |
    | Cohen's kappa | -0.2069 |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 0.22 ms |

## Improvements

The deterministic local path passes the supplied functional and adversarial tests quickly. In production, the fallback judge and rail should be replaced or supplemented with authenticated model calls, while retaining local PII detection as the first low-latency layer. RAGAS should be run on a scheduled sample with thresholds enforced in CI. The largest operational risk is external LLM latency and availability, so caching, timeouts, and a fail-closed policy should be added around the remote rails.
