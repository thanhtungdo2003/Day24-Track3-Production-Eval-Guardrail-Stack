from __future__ import annotations

"""Phase A: deterministic RAG evaluation with an optional RAGAS-compatible backend."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ANSWERS_PATH, TEST_SET_PATH

DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return sum((self.faithfulness, self.answer_relevancy,
                    self.context_precision, self.context_recall)) / 4

    @property
    def worst_metric(self) -> str:
        values = {name: getattr(self, name) for name in DIAGNOSTIC_TREE}
        return min(values, key=values.get)


def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"answers_50q.json not found at {path}; run setup_answers.py")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    groups = {name: [] for name in ("factual", "multi_hop", "adversarial")}
    for item in test_set:
        distribution = item.get("distribution")
        if distribution not in groups:
            raise ValueError(f"Unknown distribution: {distribution}")
        groups[distribution].append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    from src.m4_eval import evaluate_ragas
    raw = evaluate_ragas(
        [item["question"] for item in answers], [item["answer"] for item in answers],
        [item.get("contexts", []) for item in answers], [item["ground_truth"] for item in answers])
    return [RagasResult(
        question_id=item["id"], distribution=item["distribution"], question=item["question"],
        answer=item["answer"], contexts=item.get("contexts", []), ground_truth=item["ground_truth"],
        faithfulness=metrics.faithfulness, answer_relevancy=metrics.answer_relevancy,
        context_precision=metrics.context_precision, context_recall=metrics.context_recall,
    ) for item, metrics in zip(answers, raw["per_question"])]


def bottom_10(results: list[RagasResult]) -> list[dict]:
    output = []
    for rank, result in enumerate(sorted(results, key=lambda item: item.avg_score)[:10], 1):
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[result.worst_metric]
        output.append({"rank": rank, "question_id": result.question_id,
                       "distribution": result.distribution, "question": result.question,
                       "avg_score": round(result.avg_score, 4),
                       "worst_metric": result.worst_metric, "diagnosis": diagnosis,
                       "suggested_fix": suggested_fix})
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    distributions = ["factual", "multi_hop", "adversarial"]
    matrix = {metric: {dist: 0 for dist in distributions} for metric in DIAGNOSTIC_TREE}
    for result in results:
        matrix[result.worst_metric][result.distribution] += 1
    dominant_dist = max(distributions, key=lambda dist: sum(matrix[m][dist] for m in matrix))
    dominant_metric = max(matrix, key=lambda metric: sum(matrix[metric].values()))
    insight = (f"Distribution '{dominant_dist}' có nhiều failure nhất. "
               f"Metric '{dominant_metric}' là điểm yếu chủ đạo. "
               f"Gợi ý: {DIAGNOSTIC_TREE[dominant_metric][1]}")
    return {"matrix": matrix, "dominant_failure_distribution": dominant_dist,
            "dominant_failure_metric": dominant_metric, "insight": insight}


def save_phase_a_report(results: list[RagasResult], clusters: dict,
                        path: str = "reports/ragas_50q.json") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    per_dist = {}
    for dist in ("factual", "multi_hop", "adversarial"):
        subset = [result for result in results if result.distribution == dist]
        if subset:
            per_dist[dist] = {"count": len(subset), **{
                metric: round(sum(getattr(item, metric) for item in subset) / len(subset), 4)
                for metric in DIAGNOSTIC_TREE},
                "avg_score": round(sum(item.avg_score for item in subset) / len(subset), 4)}
    report = {"total_questions": len(results), "per_distribution": per_dist,
              "failure_clusters": clusters, "bottom_10": bottom_10(results)}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    values = run_ragas_50q(load_answers())
    save_phase_a_report(values, cluster_analysis(values))
    print(f"Phase A report saved: {len(values)} questions")
