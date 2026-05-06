#!/usr/bin/env python3
"""
SGDAgent QA Accuracy Evaluator (ported from DVD evaluate_results.py).

核心口径:
1) 以 benchmark_results.json 为主输入（避免并行 per_video 日志交织带来的低估）
2) 若存在 traces/*.jsonl，则优先使用 answer_check.is_correct 作为判定来源
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sgd.path_config import eval_output_root, lvbench_root


DEFAULT_META_PATH = str(lvbench_root() / "video_info.meta.jsonl")
DEFAULT_EVAL_OUTPUT_ROOT = eval_output_root()


def normalize_option_answer(answer: Any) -> Optional[str]:
    if answer is None:
        return None
    value = str(answer).strip()
    if not value:
        return None
    value = value.strip("`'\" ").upper()
    if re.fullmatch(r"[A-D]", value):
        return value

    for pattern in (r"\(([A-D])\)", r"\b([A-D])\b", r"^([A-D])[\.\):\s]"):
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return value


def coerce_optional_bool(value: Any) -> Optional[bool]:
    if value is True:
        return True
    if value is False:
        return False
    return None


def derive_is_correct(predicted_answer: Any, expected_answer: Any) -> Optional[bool]:
    predicted = normalize_option_answer(predicted_answer)
    expected = normalize_option_answer(expected_answer)
    if predicted is None or expected is None:
        return None
    return predicted == expected


def load_meta_file(meta_path: Path) -> Dict[str, Dict[str, Any]]:
    meta_dict: Dict[str, Dict[str, Any]] = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            key = str(data.get("key", "")).strip()
            if key:
                meta_dict[key] = data
    return meta_dict


def build_uid_index(meta_dict: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    uid_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for video_key, video_data in meta_dict.items():
        for qa in video_data.get("qa", []):
            uid = qa.get("uid")
            if uid is None:
                continue
            uid_str = str(uid)
            uid_index[uid_str].append(
                {
                    "video_key": video_key,
                    "video_data": video_data,
                    "qa": qa,
                }
            )
    return uid_index


def normalize_question_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def choose_meta_entry(entries: List[Dict[str, Any]], benchmark_question: Any) -> Optional[Dict[str, Any]]:
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0]

    bench_q = normalize_question_text(benchmark_question)
    if bench_q:
        for entry in entries:
            qa_q = normalize_question_text(entry["qa"].get("question"))
            if qa_q == bench_q:
                return entry
    return entries[0]


def load_benchmark_results(benchmark_results_path: Path) -> Dict[str, Dict[str, Any]]:
    with open(benchmark_results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"benchmark_results is not a dict: {benchmark_results_path}")

    normalized: Dict[str, Dict[str, Any]] = {}
    for qid, row in data.items():
        qid_str = str(qid)
        normalized[qid_str] = row if isinstance(row, dict) else {}
    return normalized


def load_trace_answer_checks(trace_dir: Path) -> Dict[str, Dict[str, Any]]:
    trace_checks: Dict[str, Dict[str, Any]] = {}
    if not trace_dir.exists():
        return trace_checks

    for trace_file in sorted(trace_dir.glob("*.jsonl")):
        last_answer_check = None
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event_type") == "answer_check":
                    last_answer_check = event

        if last_answer_check is None:
            trace_checks[trace_file.stem] = {
                "is_correct": None,
                "predicted_answer": None,
                "expected_answer": None,
            }
            continue

        predicted = last_answer_check.get("predicted_answer_normalized")
        if predicted is None:
            predicted = normalize_option_answer(last_answer_check.get("predicted_answer_raw"))
        expected = last_answer_check.get("expected_answer_normalized")
        if expected is None:
            expected = normalize_option_answer(last_answer_check.get("expected_answer_raw"))

        trace_checks[trace_file.stem] = {
            "is_correct": coerce_optional_bool(last_answer_check.get("is_correct")),
            "predicted_answer": predicted,
            "expected_answer": expected,
        }

    return trace_checks


def infer_run_paths(
    run_input_path: Path,
    benchmark_results_arg: Optional[str],
    trace_dir_arg: Optional[str],
) -> Tuple[Path, Path, Path, Path]:
    resolved_input = run_input_path.expanduser().resolve()

    if resolved_input.name == "per_video":
        run_dir = resolved_input.parent
        per_video_dir = resolved_input
    else:
        run_dir = resolved_input
        per_video_dir = resolved_input / "per_video"

    if benchmark_results_arg:
        benchmark_results = Path(benchmark_results_arg).expanduser().resolve()
    else:
        benchmark_results = run_dir / "benchmark_results.json"

    if trace_dir_arg:
        trace_dir = Path(trace_dir_arg).expanduser().resolve()
    else:
        trace_dir = run_dir / "traces"

    return run_dir, per_video_dir, benchmark_results, trace_dir


def infer_meta_path(run_dir: Path, meta_arg: Optional[str]) -> Path:
    if meta_arg:
        candidate = Path(meta_arg).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"meta file not found: {candidate}")
        return candidate

    candidates: List[Path] = []
    subset_meta = run_dir / "subset_meta.jsonl"
    candidates.append(subset_meta)

    run_config = run_dir / "run_config.txt"
    if run_config.exists():
        try:
            with open(run_config, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Meta File:"):
                        raw = line.split(":", 1)[1].strip()
                        if raw:
                            candidates.append(Path(raw).expanduser().resolve())
                        break
        except OSError:
            pass

    candidates.append(Path(DEFAULT_META_PATH).expanduser().resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "meta file not found. Tried: " + ", ".join(str(p) for p in candidates)
    )


def qid_sort_key(qid: str) -> Tuple[int, Any]:
    if qid.isdigit():
        return (0, int(qid))
    return (1, qid)


def summarize_trace_metrics(trace_checks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    total = len(trace_checks)
    correct = sum(1 for info in trace_checks.values() if info.get("is_correct") is True)
    wrong = sum(1 for info in trace_checks.values() if info.get("is_correct") is False)
    unknown = total - correct - wrong
    answered = correct + wrong
    return {
        "total_traces": total,
        "answered": answered,
        "correct": correct,
        "wrong": wrong,
        "unknown": unknown,
        "answered_accuracy": (correct / answered * 100) if answered else 0.0,
        "e2e_accuracy": (correct / total * 100) if total else 0.0,
    }


def evaluate_from_benchmark(
    run_dir: Path,
    benchmark_results_path: Path,
    trace_dir: Path,
    meta_path: Path,
    output_file: Optional[Path],
) -> Dict[str, Any]:
    print("=" * 80)
    print("DVD Benchmark Evaluation (benchmark_results + trace)")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Benchmark results: {benchmark_results_path}")
    print(f"Trace directory: {trace_dir}")
    print(f"Meta file: {meta_path}")
    print()

    print("Loading meta file...")
    meta_dict = load_meta_file(meta_path)
    uid_index = build_uid_index(meta_dict)
    print(f"✓ Loaded {len(meta_dict)} videos from meta file")

    print("Loading benchmark results...")
    benchmark_rows = load_benchmark_results(benchmark_results_path)
    print(f"✓ Loaded {len(benchmark_rows)} question rows from benchmark_results.json")

    print("Loading trace answer_check events...")
    trace_checks = load_trace_answer_checks(trace_dir)
    print(f"✓ Loaded {len(trace_checks)} trace files")
    print()

    trace_by_qid: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for trace_id, trace_info in trace_checks.items():
        if "__" not in trace_id:
            continue
        _, qid = trace_id.rsplit("__", 1)
        trace_by_qid[qid].append((trace_id, trace_info))

    per_video_results: Dict[str, Dict[str, Any]] = {}
    unmatched_qids: List[str] = []

    for qid in sorted(benchmark_rows.keys(), key=qid_sort_key):
        row = benchmark_rows[qid]
        meta_entry = choose_meta_entry(uid_index.get(qid, []), row.get("question"))

        if meta_entry is None:
            unmatched_qids.append(qid)
            video_key = "__unmatched__"
            video_data = {"type": "unknown", "video_info": {"duration_minutes": 0}}
            qa = {
                "uid": qid,
                "question": row.get("question") or f"[unmatched qid={qid}]",
                "answer": row.get("expected_answer"),
            }
            trace_id = None
        else:
            video_key = meta_entry["video_key"]
            video_data = meta_entry["video_data"]
            qa = meta_entry["qa"]
            trace_id = f"{video_key}__{qid}"

        if video_key not in per_video_results:
            per_video_results[video_key] = {
                "video_key": video_key,
                "video_type": video_data.get("type", "unknown"),
                "duration": video_data.get("video_info", {}).get("duration_minutes", 0),
                "details": [],
            }

        predicted_answer = row.get("answer")
        expected_answer = qa.get("answer", row.get("expected_answer"))
        trace_info = None
        trace_id_used = trace_id

        if trace_id and trace_id in trace_checks:
            trace_info = trace_checks[trace_id]
        elif len(trace_by_qid.get(qid, [])) == 1:
            trace_id_used, trace_info = trace_by_qid[qid][0]

        is_correct = None
        verdict_source = "unknown"

        if trace_info is not None:
            trace_verdict = coerce_optional_bool(trace_info.get("is_correct"))
            if trace_verdict is not None:
                is_correct = trace_verdict
                verdict_source = "trace.answer_check"
            if predicted_answer is None:
                predicted_answer = trace_info.get("predicted_answer")
            if expected_answer is None:
                expected_answer = trace_info.get("expected_answer")

        if is_correct is None:
            benchmark_verdict = coerce_optional_bool(row.get("is_correct"))
            if benchmark_verdict is not None:
                is_correct = benchmark_verdict
                verdict_source = "benchmark_results.is_correct"

        if is_correct is None:
            derived_verdict = derive_is_correct(predicted_answer, expected_answer)
            if derived_verdict is not None:
                is_correct = derived_verdict
                verdict_source = "derived_from_answer"

        if is_correct is True:
            status = "CORRECT"
        elif is_correct is False:
            status = "WRONG"
        else:
            status = "UNKNOWN"

        per_video_results[video_key]["details"].append(
            {
                "uid": str(qa.get("uid", qid)),
                "question": qa.get("question", row.get("question", "")),
                "correct_answer": expected_answer,
                "predicted_answer": predicted_answer,
                "status": status,
                "is_correct": is_correct,
                "verdict_source": verdict_source,
                "trace_id": trace_id_used,
            }
        )

    all_results: List[Dict[str, Any]] = []
    type_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "wrong": 0, "unknown": 0, "answered": 0, "total": 0}
    )

    total_correct = 0
    total_wrong = 0
    total_unknown = 0

    video_keys = sorted(per_video_results.keys(), key=lambda x: (x == "__unmatched__", x))
    for video_key in video_keys:
        payload = per_video_results[video_key]
        details = payload["details"]
        correct = sum(1 for d in details if d.get("is_correct") is True)
        wrong = sum(1 for d in details if d.get("is_correct") is False)
        unknown = len(details) - correct - wrong
        answered = correct + wrong
        total = len(details)

        payload["correct"] = correct
        payload["wrong"] = wrong
        payload["unknown"] = unknown
        payload["answered"] = answered
        payload["total"] = total
        payload["accuracy"] = (correct / total * 100) if total else 0.0
        payload["answered_accuracy"] = (correct / answered * 100) if answered else 0.0

        all_results.append(payload)

        total_correct += correct
        total_wrong += wrong
        total_unknown += unknown

        vtype = payload["video_type"]
        type_stats[vtype]["correct"] += correct
        type_stats[vtype]["wrong"] += wrong
        type_stats[vtype]["unknown"] += unknown
        type_stats[vtype]["answered"] += answered
        type_stats[vtype]["total"] += total

    total_questions = len(benchmark_rows)
    answered_questions = total_correct + total_wrong
    overall_accuracy = (total_correct / total_questions * 100) if total_questions else 0.0
    answered_accuracy = (total_correct / answered_questions * 100) if answered_questions else 0.0

    trace_summary = summarize_trace_metrics(trace_checks)

    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Videos evaluated: {len(all_results)}")
    print(f"Total questions: {total_questions}")
    print(f"Answered questions: {answered_questions}")
    print(f"Unknown questions: {total_unknown}")
    print(f"Correct answers: {total_correct}")
    print(f"Wrong answers: {total_wrong}")
    print(f"E2E accuracy: {overall_accuracy:.2f}%")
    print(f"Answered accuracy: {answered_accuracy:.2f}%")
    print()

    if unmatched_qids:
        print(f"Warning: {len(unmatched_qids)} qids not found in meta (showing first 10): {unmatched_qids[:10]}")

    if trace_summary["total_traces"] > 0:
        print("Trace summary:")
        print(
            f"  total={trace_summary['total_traces']}, "
            f"correct={trace_summary['correct']}, wrong={trace_summary['wrong']}, unknown={trace_summary['unknown']}, "
            f"e2e={trace_summary['e2e_accuracy']:.2f}%"
        )
        if trace_summary["total_traces"] != total_questions:
            print(
                "  Warning: trace total does not match benchmark_results total. "
                "Please confirm run directory inputs."
            )
        print()

    print("Accuracy by video type:")
    for vtype in sorted(type_stats.keys()):
        stats = type_stats[vtype]
        total = stats["total"]
        answered = stats["answered"]
        e2e = (stats["correct"] / total * 100) if total else 0.0
        ans_acc = (stats["correct"] / answered * 100) if answered else 0.0
        print(
            f"  {vtype:12s}: "
            f"{stats['correct']:3d}/{total:3d} e2e={e2e:5.1f}% "
            f"(answered_acc={ans_acc:5.1f}%)"
        )
    print()

    type_stats_output: Dict[str, Dict[str, Any]] = {}
    for vtype, stats in type_stats.items():
        total = stats["total"]
        answered = stats["answered"]
        type_stats_output[vtype] = {
            "correct": stats["correct"],
            "wrong": stats["wrong"],
            "unknown": stats["unknown"],
            "answered": answered,
            "total": total,
            "overall_accuracy": (stats["correct"] / total * 100) if total else 0.0,
            "answered_accuracy": (stats["correct"] / answered * 100) if answered else 0.0,
        }

    output_payload = {
        "summary": {
            "run_dir": str(run_dir),
            "benchmark_results": str(benchmark_results_path),
            "trace_dir": str(trace_dir),
            "meta_file": str(meta_path),
            "videos_evaluated": len(all_results),
            "total_questions": total_questions,
            "answered_questions": answered_questions,
            "unknown_questions": total_unknown,
            "total_correct": total_correct,
            "total_wrong": total_wrong,
            "overall_accuracy": overall_accuracy,
            "answered_accuracy": answered_accuracy,
            "score_source": "trace.answer_check -> benchmark_results.is_correct -> derived(answer,expected)",
            "trace_summary": trace_summary,
            "type_stats": type_stats_output,
            "unmatched_qids_count": len(unmatched_qids),
            "unmatched_qids_sample": unmatched_qids[:50],
        },
        "results": all_results,
    }

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2, ensure_ascii=False)
        print(f"✓ Detailed results saved to: {output_file}")

    return output_payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SGDAgent benchmark results using benchmark_results.json "
            "and optional trace answer_check events."
        )
    )
    parser.add_argument(
        "run_path",
        help="Run directory or legacy per_video directory path.",
    )
    parser.add_argument(
        "--benchmark-results",
        default=None,
        help="Path to benchmark_results.json. Default: <run-dir>/benchmark_results.json",
    )
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="Path to traces directory. Default: <run-dir>/traces",
    )
    parser.add_argument(
        "--meta",
        default=None,
        help=(
            "Path to meta file. Default auto-detect order: "
            "<run-dir>/subset_meta.jsonl -> Meta File in run_config.txt -> default LVBench meta."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON file for detailed results.",
    )
    args = parser.parse_args()

    run_input = Path(args.run_path)
    if not run_input.exists():
        print(f"Error: run path not found: {run_input}")
        return 1

    run_dir, _per_video_dir, benchmark_results_path, trace_dir = infer_run_paths(
        run_input_path=run_input,
        benchmark_results_arg=args.benchmark_results,
        trace_dir_arg=args.trace_dir,
    )

    try:
        meta_path = infer_meta_path(run_dir=run_dir, meta_arg=args.meta)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    if not benchmark_results_path.exists():
        print(f"Error: benchmark_results.json not found: {benchmark_results_path}")
        print("Hint: pass --benchmark-results <path> or provide the run directory as positional arg.")
        return 1

    if args.output:
        output_file = Path(args.output).expanduser().resolve()
    else:
        output_file = DEFAULT_EVAL_OUTPUT_ROOT / f"results_{run_dir.name}.json"

    evaluate_from_benchmark(
        run_dir=run_dir,
        benchmark_results_path=benchmark_results_path,
        trace_dir=trace_dir,
        meta_path=meta_path,
        output_file=output_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
