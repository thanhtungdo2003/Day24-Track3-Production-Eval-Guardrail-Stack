from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QuestionMetrics:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"\w+", value.lower()))


def evaluate_ragas(questions, answers, contexts, ground_truths):
    per_question = []
    for question, answer, context, truth in zip(questions, answers, contexts, ground_truths):
        answer_words, truth_words = _tokens(answer), _tokens(truth)
        question_words = _tokens(question)
        context_words = _tokens(" ".join(context))
        overlap = len(answer_words & truth_words) / max(len(truth_words), 1)
        relevance = len(answer_words & question_words) / max(len(question_words), 1)
        recall = len(truth_words & context_words) / max(len(truth_words), 1)
        precision = len(answer_words & context_words) / max(len(answer_words), 1)
        per_question.append(QuestionMetrics(*(round(max(0.0, min(1.0, value)), 4)
                                             for value in (overlap, relevance, precision, recall))))
    return {"per_question": per_question}

