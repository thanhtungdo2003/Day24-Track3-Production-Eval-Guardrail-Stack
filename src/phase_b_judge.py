from __future__ import annotations

"""Phase B: pairwise judging with optional OpenAI integration and local fallback."""

import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HUMAN_LABELS_PATH, NVIDIA_LIVE_JUDGE, NVIDIA_MODEL, get_llm_client


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str
    winner_pass2: str
    final_winner: str
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool
    scores_pass1: dict = field(default_factory=dict)
    scores_pass2: dict = field(default_factory=dict)


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"\w+", value.lower()))


def _local_judge(question: str, answer_a: str, answer_b: str) -> dict:
    question_words = _tokens(question)
    scores = {}
    for label, answer in (("A", answer_a), ("B", answer_b)):
        words = _tokens(answer)
        relevance = len(words & question_words) / max(len(question_words), 1)
        scores[label] = round(min(1.0, 0.7 * relevance + 0.3 * min(1.0, len(words) / 20)), 3)
    if abs(scores["A"] - scores["B"]) < 0.05:
        winner = "tie"
    else:
        winner = max(scores, key=scores.get)
    return {"winner": winner, "reasoning": "Local judge compared question overlap and completeness.", "scores": scores}


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    client = get_llm_client() if NVIDIA_LIVE_JUDGE else None
    if client:
        try:
            prompt = (f"Question: {question}\nA: {answer_a}\nB: {answer_b}\n"
                      "Return JSON with winner A, B, or tie, reasoning, and scores A/B in [0,1].")
            response = client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[{"role": "system", "content": "You are a precise RAG answer judge. JSON only."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0.0,
                max_tokens=512,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            result = json.loads(response.choices[0].message.content)
            if result.get("winner") in {"A", "B", "tie"}:
                return result
        except Exception:
            pass
    return _local_judge(question, answer_a, answer_b)


def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    first = pairwise_judge(question, answer_a, answer_b)
    second_raw = pairwise_judge(question, answer_b, answer_a)
    winner_second = {"A": "B", "B": "A", "tie": "tie"}[second_raw["winner"]]
    final = first["winner"] if first["winner"] == winner_second else "tie"
    return JudgeResult(question, answer_a, answer_b, first["winner"], winner_second, final,
                       first.get("reasoning", ""), second_raw.get("reasoning", ""),
                       first["winner"] == winner_second, first.get("scores", {}),
                       {"A": second_raw.get("scores", {}).get("B", 0.0),
                        "B": second_raw.get("scores", {}).get("A", 0.0)})


def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    if len(judge_labels) != len(human_labels) or not judge_labels:
        return 0.0
    n = len(judge_labels)
    observed = sum(a == b for a, b in zip(judge_labels, human_labels)) / n
    expected = sum((judge_labels.count(label) / n) * (human_labels.count(label) / n)
                   for label in (0, 1))
    return round((observed - expected) / (1 - expected), 6) if expected != 1 else 1.0


def bias_report(judge_results: list[JudgeResult]) -> dict:
    total = len(judge_results)
    if not total:
        return {"total_judged": 0, "position_bias_rate": 0.0, "position_bias_count": 0,
                "verbosity_bias": 0.0, "verbosity_details": {"a_wins_a_longer": 0,
                "b_wins_b_longer": 0, "total_decisive": 0}, "interpretation": "No judgments."}
    position_count = sum(not item.position_consistent for item in judge_results)
    a_longer = sum(item.final_winner == "A" and len(item.answer_a) > len(item.answer_b) for item in judge_results)
    b_longer = sum(item.final_winner == "B" and len(item.answer_b) > len(item.answer_a) for item in judge_results)
    decisive = sum(item.final_winner != "tie" for item in judge_results)
    verbosity = (a_longer + b_longer) / decisive if decisive else 0.0
    return {"total_judged": total, "position_bias_rate": round(position_count / total, 3),
            "position_bias_count": position_count, "verbosity_bias": round(verbosity, 3),
            "verbosity_details": {"a_wins_a_longer": a_longer, "b_wins_b_longer": b_longer,
                                   "total_decisive": decisive},
            "interpretation": "Position bias high; retain swap-and-average." if position_count / total > .3
            else "Position bias low; judge is stable."}


if __name__ == "__main__":
    from pathlib import Path
    from config import HUMAN_LABELS_PATH, TEST_SET_PATH

    human = json.loads(Path(HUMAN_LABELS_PATH).read_text(encoding="utf-8"))
    test_set = {item["id"]: item for item in json.loads(Path(TEST_SET_PATH).read_text(encoding="utf-8"))}
    results, judge_labels, human_labels = [], [], []
    for item in human:
        result = swap_and_average(item["question"], item["model_answer"],
                                  test_set[item["question_id"]]["ground_truth"])
        results.append(result)
        judge_labels.append(1 if result.final_winner == "A" else 0)
        human_labels.append(item["human_label"])
    report = bias_report(results)
    report.update({"cohen_kappa": cohen_kappa(judge_labels, human_labels),
                   "results": [result.__dict__ for result in results]})
    Path("reports").mkdir(exist_ok=True)
    Path("reports/judge_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Phase B report saved: {len(results)} judgments; kappa={report['cohen_kappa']:.6f}")
