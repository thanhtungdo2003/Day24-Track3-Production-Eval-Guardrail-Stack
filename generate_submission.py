from __future__ import annotations

import json
from pathlib import Path

from src.phase_a_ragas import cluster_analysis, run_ragas_50q, save_phase_a_report
from src.phase_b_judge import bias_report, cohen_kappa, swap_and_average
from src.phase_c_guard import measure_p95_latency, run_adversarial_suite


ROOT = Path(__file__).parent


def main() -> None:
    test_set = json.loads((ROOT / "test_set_50q.json").read_text(encoding="utf-8"))
    answers_path = ROOT / "answers_50q.json"
    if not answers_path.exists():
        raise FileNotFoundError("answers_50q.json is missing; run `uv run python setup_answers.py` first")
    answers = json.loads(answers_path.read_text(encoding="utf-8"))
    if len(answers) != 50:
        raise ValueError(f"Expected 50 generated answers, found {len(answers)}")
    ragas = run_ragas_50q(answers)
    save_phase_a_report(ragas, cluster_analysis(ragas))

    human = json.loads((ROOT / "human_labels_10q.json").read_text(encoding="utf-8"))
    ground_truth_by_id = {item["id"]: item["ground_truth"] for item in test_set}
    judge_results = []
    judge_labels = []
    for item in human:
        judged = swap_and_average(item["question"], item["model_answer"], ground_truth_by_id[item["question_id"]])
        judge_results.append(judged)
        judge_labels.append(1 if judged.final_winner == "A" else 0)
    judge_report = bias_report(judge_results)
    judge_report.update({"cohen_kappa": cohen_kappa(judge_labels, [item["human_label"] for item in human]),
                         "results": [result.__dict__ for result in judge_results]})
    (ROOT / "reports/judge_results.json").write_text(json.dumps(judge_report, ensure_ascii=False, indent=2), encoding="utf-8")

    adversarial = json.loads((ROOT / "adversarial_set_20.json").read_text(encoding="utf-8"))
    guard_results = run_adversarial_suite(adversarial)
    latency = measure_p95_latency([item["input"] for item in adversarial], n_runs=20)
    guard_report = {"adversarial_results": guard_results,
                    "passed": sum(item["passed"] for item in guard_results),
                    "total": len(guard_results), "pass_rate": sum(item["passed"] for item in guard_results) / len(guard_results),
                    "latency": latency}
    (ROOT / "reports/guard_results.json").write_text(json.dumps(guard_report, ensure_ascii=False, indent=2), encoding="utf-8")

    averages = {dist: sum(item.avg_score for item in ragas if item.distribution == dist) /
                len([item for item in ragas if item.distribution == dist]) for dist in ("factual", "multi_hop", "adversarial")}
    phase_a = json.loads((ROOT / "reports/ragas_50q.json").read_text(encoding="utf-8"))
    blueprint = f"""# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Student:** Lab submission
**Date:** 2026-08-26

## Guard Stack Architecture

User input flows through local PII scanning, the input rail, the Lab 18 RAG pipeline, and the output rail. PII is anonymized or blocked before retrieval; jailbreak, prompt-injection, and off-topic requests are refused; sensitive output is replaced with a safe response.

## Latency Budget

| Layer | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---:|---:|---:|
| Presidio/regex PII | {latency['presidio_ms']['p50']} | {latency['presidio_ms']['p95']} | {latency['presidio_ms']['p99']} |
| NeMo/input fallback | {latency['nemo_ms']['p50']} | {latency['nemo_ms']['p95']} | {latency['nemo_ms']['p99']} |
| Total guard | {latency['total_ms']['p50']} | {latency['total_ms']['p95']} | {latency['total_ms']['p99']} |

**Budget OK:** {'Yes' if latency['latency_budget_ok'] else 'No'}; target is < {latency['budget_ms']} ms P95.

## CI/CD Gates

Run `uv run pytest tests/`, `uv run python src/phase_a_ragas.py`, and `uv run python generate_submission.py`. Merge only when all tests pass, the RAG report contains 50 questions, adversarial pass rate is at least 75%, and guard P95 remains below 500 ms.

## Monitoring Dashboard

Alert when daily faithfulness is below 0.70, adversarial block rate is below 80%, guard P95 exceeds 600 ms, or PII detections spike above 10 per hour. Investigate the corresponding report and review new attack patterns before changing thresholds.

## Results

| Metric | Result |
|---|---:|
| RAGAS avg score | {sum(item.avg_score for item in ragas) / len(ragas):.4f} |
| Factual / multi-hop / adversarial average | {averages['factual']:.4f} / {averages['multi_hop']:.4f} / {averages['adversarial']:.4f} |
| Dominant failure distribution | {phase_a['failure_clusters']['dominant_failure_distribution']} |
    | Cohen's kappa | {judge_report['cohen_kappa']:.4f} |
| Adversarial pass rate | {guard_report['passed']} / {guard_report['total']} |
| Guard P95 latency | {latency['total_ms']['p95']} ms |

## Improvements

The deterministic local path passes the supplied functional and adversarial tests quickly. In production, the fallback judge and rail should be replaced or supplemented with authenticated model calls, while retaining local PII detection as the first low-latency layer. RAGAS should be run on a scheduled sample with thresholds enforced in CI. The largest operational risk is external LLM latency and availability, so caching, timeouts, and a fail-closed policy should be added around the remote rails.
"""
    (ROOT / "reports/blueprint.md").write_text(blueprint, encoding="utf-8")
    print(f"Generated answers, Phase A/B/C reports, and blueprint ({len(ragas)} questions; {guard_report['passed']}/{guard_report['total']} attacks passed).")


if __name__ == "__main__":
    main()
