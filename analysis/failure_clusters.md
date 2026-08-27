# Failure Cluster Analysis - Phase A

**Date:** 2026-08-27
**Evaluation:** 50 generated answers retrieved from the repository `data/` corpus

## Aggregate Scores

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.6911 | 0.4457 | 0.3351 |
| answer_relevancy | 0.6400 | 0.5095 | 0.7129 |
| context_precision | 1.0000 | 1.0000 | 1.0000 |
| context_recall | 0.8408 | 0.6295 | 0.5032 |
| **avg_score** | **0.7930** | **0.6462** | **0.6378** |

The evaluator uses retrieved corpus text and a deterministic local metric implementation because no OpenAI/RAGAS service is configured. The lower multi-hop and adversarial scores indicate that simple lexical retrieval is not sufficient for cross-document reasoning and version conflicts.

## Bottom 10

| Rank | ID | Distribution | Average | Worst metric |
|---:|---:|---|---:|---|
| 1 | 40 | multi_hop | 0.4456 | faithfulness |
| 2 | 50 | adversarial | 0.4472 | faithfulness |
| 3 | 22 | multi_hop | 0.5232 | faithfulness |
| 4 | 37 | multi_hop | 0.5307 | answer_relevancy |
| 5 | 41 | adversarial | 0.5525 | faithfulness |
| 6 | 33 | multi_hop | 0.5564 | faithfulness |
| 7 | 34 | multi_hop | 0.5566 | faithfulness |
| 8 | 21 | multi_hop | 0.5598 | faithfulness |
| 9 | 49 | adversarial | 0.5762 | faithfulness |
| 10 | 27 | multi_hop | 0.5955 | faithfulness |

## Failure Cluster Matrix

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 8 | 13 | 10 | 31 |
| answer_relevancy | 12 | 7 | 0 | 19 |
| context_precision | 0 | 0 | 0 | 0 |
| context_recall | 0 | 0 | 0 | 0 |

**Dominant failure metric:** faithfulness, with 31 of 50 questions. The raw count selects factual as the dominant distribution only because it has 20 questions; by rate, adversarial is weakest at 100% faithfulness failures and multi-hop is next at 65%.

The main issue is retrieval-only answering: a selected chunk can be relevant but still omit the second document, current policy version, calculation, or exception needed by the question. The production fix is to restore the Day 18 dense/hybrid search and reranker, then add explicit multi-hop context assembly and current-version filtering.

## Suggested Fixes

| Metric | Root cause | Suggested fix |
|---|---|---|
| faithfulness | Retrieved text is returned without grounded synthesis | Use a grounded LLM answer with citations and contradiction checks |
| context_recall | One lexical chunk misses cross-document evidence | Use dense + BM25 hybrid retrieval and increase candidate depth |
| context_precision | Candidate set contains distractors | Apply the Day 18 reranker and metadata/version filters |
| answer_relevancy | Retrieved prose does not directly answer the question | Add an answer-focused prompt and semantic relevance evaluation |
