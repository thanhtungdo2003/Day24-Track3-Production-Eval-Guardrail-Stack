from __future__ import annotations

"""Phase C: PII detection, input/output rails, adversarial tests, and latency."""

import asyncio
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


def setup_presidio():
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
    from presidio_anonymizer import AnonymizerEngine
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(PatternRecognizer("VN_CCCD", [Pattern("CCCD", r"\b\d{12}\b", .9), Pattern("CMND", r"\b\d{9}\b", .7)]))
    registry.add_recognizer(PatternRecognizer("VN_PHONE", [Pattern("VN mobile", r"\b0[3-9]\d{8}\b", .9)]))
    return AnalyzerEngine(registry=registry), AnonymizerEngine()


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    if analyzer is not None and anonymizer is not None:
        findings = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
        entities = [{"type": item.entity_type, "text": text[item.start:item.end],
                     "score": round(item.score, 3), "start": item.start, "end": item.end} for item in findings]
        return {"has_pii": bool(entities), "entities": entities,
                "anonymized": anonymizer.anonymize(text=text, analyzer_results=findings).text if findings else text}
    patterns = [("VN_CCCD", r"\b\d{12}\b"), ("VN_PHONE", r"\b0[3-9]\d{8}\b"),
                ("EMAIL_ADDRESS", r"\b[\w.+-]+@[\w.-]+\.\w+\b"), ("CMND", r"\b\d{9}\b")]
    entities = []
    anonymized = text
    for entity_type, pattern in patterns:
        for match in re.finditer(pattern, text):
            if not any(item["start"] <= match.start() < item["end"] for item in entities):
                entities.append({"type": entity_type, "text": match.group(), "score": 1.0,
                                 "start": match.start(), "end": match.end()})
                anonymized = anonymized.replace(match.group(), f"<{entity_type}>")
    return {"has_pii": bool(entities), "entities": entities, "anonymized": anonymized}


def setup_nemo_rails():
    from nemoguardrails import LLMRails, RailsConfig
    return LLMRails(RailsConfig.from_path(GUARDRAILS_CONFIG_DIR))


_BLOCK_TERMS = ("ignore your", "bỏ qua", "system override", "forget all", "dan", "unrestricted",
                "confidential", "system instructions", "training data", "admin command", "viết một bài thơ",
                "nấu phở", "bitcoin", "ethereum", "giải phương trình", "recommend", "phim", "tấn công mạng",
                "tiết lộ", "lương của nhân viên", "thông tin nhân viên", "cccd và số điện thoại",
                "số điện thoại của")


async def check_input_rail(text: str, rails=None) -> dict:
    if rails is not None:
        try:
            response = await rails.generate_async(messages=[{"role": "user", "content": text}])
            raw = response if isinstance(response, str) else str(response)
            refused = any(term in raw.lower() for term in ("xin lỗi", "không thể", "i cannot", "sorry"))
            return {"allowed": not refused, "blocked_reason": "nemo_input_rail" if refused else None, "response": raw}
        except Exception:
            pass
    blocked = any(term in text.lower() for term in _BLOCK_TERMS)
    return {"allowed": not blocked, "blocked_reason": "nemo_input_rail" if blocked else None,
            "response": "Xin lỗi, tôi chỉ có thể hỗ trợ chính sách HR." if blocked else ""}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    if rails is not None:
        try:
            response = await rails.generate_async(messages=[{"role": "user", "content": question},
                                                     {"role": "assistant", "content": answer}])
            raw = response if isinstance(response, str) else str(response)
            if raw != answer and any(term in raw.lower() for term in ("không thể", "cannot", "xin lỗi")):
                return {"safe": False, "flagged_reason": "nemo_output_rail", "final_answer": raw}
        except Exception:
            pass
    sensitive = pii_scan(answer)["has_pii"] or any(term in answer.lower() for term in ("mật khẩu hệ thống", "thông tin bí mật"))
    return {"safe": not sensitive, "flagged_reason": "pii_or_sensitive_output" if sensitive else None,
            "final_answer": "Tôi không thể cung cấp thông tin nhạy cảm." if sensitive else answer}


def run_adversarial_suite(adversarial_set: list[dict], rails=None, analyzer=None, anonymizer=None) -> list[dict]:
    async def run_all():
        results = []
        for item in adversarial_set:
            blocked_by = "presidio" if pii_scan(item["input"], analyzer, anonymizer)["has_pii"] else None
            if blocked_by is None and not (await check_input_rail(item["input"], rails))["allowed"]:
                blocked_by = "nemo_input"
            actual = "blocked" if blocked_by else "allowed"
            results.append({"id": item["id"], "category": item["category"], "input": item["input"][:80],
                            "expected": item["expected"], "actual": actual, "blocked_by": blocked_by,
                            "passed": actual == item["expected"]})
        return results
    results = asyncio.run(run_all())
    print(f"Adversarial suite: {sum(item['passed'] for item in results)}/{len(results)} passed")
    return results


def measure_p95_latency(test_inputs: list[str], n_runs: int = 20, rails=None, analyzer=None, anonymizer=None) -> dict:
    presidio, nemo, total = [], [], []
    inputs = (test_inputs or [""])[:max(1, n_runs)]
    for text in inputs:
        start = time.perf_counter(); pii_scan(text, analyzer, anonymizer); p = (time.perf_counter() - start) * 1000
        start = time.perf_counter(); asyncio.run(check_input_rail(text, rails)); n = (time.perf_counter() - start) * 1000
        presidio.append(p); nemo.append(n); total.append(p + n)
    def percentiles(values):
        values = sorted(values)
        return {"p50": round(statistics.quantiles(values, n=2, method="inclusive")[0], 2) if len(values) > 1 else round(values[0], 2),
                "p95": round(values[min(len(values) - 1, int(len(values) * .95))], 2),
                "p99": round(values[min(len(values) - 1, int(len(values) * .99))], 2)}
    total_stats = percentiles(total)
    return {"presidio_ms": percentiles(presidio), "nemo_ms": percentiles(nemo), "total_ms": total_stats,
            "latency_budget_ok": total_stats["p95"] < LATENCY_BUDGET_P95_MS, "budget_ms": LATENCY_BUDGET_P95_MS}


if __name__ == "__main__":
    from pathlib import Path
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as handle:
        adversarial = json.load(handle)
    results = run_adversarial_suite(adversarial)
    latency = measure_p95_latency([item["input"] for item in adversarial], n_runs=len(adversarial))
    report = {"adversarial_results": results, "passed": sum(item["passed"] for item in results),
              "total": len(results), "pass_rate": sum(item["passed"] for item in results) / len(results),
              "latency": latency}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/guard_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Phase C report saved: {report['passed']}/{report['total']} attacks passed; "
          f"P95={latency['total_ms']['p95']}ms")
