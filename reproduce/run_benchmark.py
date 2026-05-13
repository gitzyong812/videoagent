import sgd.config as config
import os
import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from sgd.sgd_core import SGDCoreAgent
from sgd.video_utils import load_video, decode_video_to_frames, download_srt_subtitle
from sgd.frame_caption import process_video, process_video_lite
from sgd.utils import extract_answer


def _normalize_option_answer(answer):
    if answer is None:
        return None
    value = str(answer).strip()
    if not value:
        return None
    value = value.strip("`'\" ").upper()

    if re.fullmatch(r"[A-D]", value):
        return value

    for pattern in [
        r"^\(?([A-D])\)?[\.\):]?$",
        r"^OPTION\s+\(?([A-D])\)?[\.\):]?$",
        r"^ANSWER\s*[:\-]?\s*\(?([A-D])\)?[\.\):]?$",
        r"(?:CORRECT\s+ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*\(?([A-D])\)?",
        r"\(([A-D])\)",
        r"(?:^|[^A-Z])\(?([A-D])\)?\s*$",
    ]:
        matches = list(re.finditer(pattern, value))
        if matches:
            return matches[-1].group(1)
    return None


def _is_single_choice_question(question):
    if not question:
        return False
    return all(token in question for token in ("(A)", "(B)", "(C)", "(D)"))


def _extract_answer_from_messages(messages, question):
    if not messages:
        return None

    if _is_single_choice_question(question):
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            candidate = extract_answer(msg)
            normalized = _normalize_option_answer(candidate)
            if normalized in {"A", "B", "C", "D"}:
                return normalized
        return None

    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        candidate = extract_answer(msg)
        if candidate:
            return candidate
    return None


def _read_last_event_idx(trace_file_path):
    if not os.path.exists(trace_file_path):
        return 0
    last_idx = 0
    with open(trace_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_idx = event.get("event_idx")
            if isinstance(event_idx, int):
                last_idx = max(last_idx, event_idx)
    return last_idx


def _append_answer_check_event(trace_file_path, trace_id, predicted_answer, expected_answer):
    predicted_norm = _normalize_option_answer(predicted_answer)
    expected_norm = _normalize_option_answer(expected_answer)
    is_correct = None
    if predicted_norm is not None and expected_norm is not None:
        is_correct = (predicted_norm == expected_norm)

    event = {
        "event_idx": _read_last_event_idx(trace_file_path) + 1,
        "event_type": "answer_check",
        "trace_id": trace_id,
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "predicted_answer_raw": predicted_answer,
        "predicted_answer_normalized": predicted_norm,
        "expected_answer_raw": expected_answer,
        "expected_answer_normalized": expected_norm,
        "is_correct": is_correct,
    }
    trace_parent = os.path.dirname(os.path.abspath(trace_file_path))
    if trace_parent:
        os.makedirs(trace_parent, exist_ok=True)
    with open(trace_file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def _log_model_config():
    print("[MODEL CONFIG]")
    print(f"  orchestrator_llm={config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME}")
    print(f"  tool_vlm={config.AOAI_TOOL_VLM_MODEL_NAME}")
    print(f"  caption_vlm={config.AOAI_CAPTION_VLM_MODEL_NAME}")
    print(f"  stage1_graph_vlm={getattr(config, 'SCENE_GRAPH_STAGE1_VLM_MODEL_NAME', '<unset>')}")


def main():
    parser = argparse.ArgumentParser(description="Run SGDCoreAgent on a video.")
    parser.add_argument("benchmark_database_folder", help="The path to the benchmark database folder.")
    parser.add_argument("benchmark_metadata", help="The path to the benchmark metadata file.")
    parser.add_argument("--per-video-log-dir", default=None,
                        help="Optional directory to write per-video logs.")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Number of concurrent questions per video. Default: 1")
    parser.add_argument("--trace-log-dir", default=None,
                        help="Directory to write step-by-step trace JSONL logs.")
    parser.add_argument(
        "--caption-root",
        default=None,
        help=(
            "Optional root folder for caption artifacts (per-video captions/). "
            "Defaults to benchmark_database_folder. "
            "Use this to keep frames/database under one root and captions/graphs under another."
        ),
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help=(
            "Path to write final benchmark results JSON. "
            "Defaults to <run_log_root>/benchmark_results.json when --per-video-log-dir is set; "
            "otherwise SGD_BENCHMARK_RESULTS_FILE or ./benchmark_results.json."
        ),
    )
    args = parser.parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")

    benchmark_database_folder = args.benchmark_database_folder
    caption_root = args.caption_root or benchmark_database_folder
    if config.TRACE_ENABLED:
        if args.trace_log_dir:
            trace_log_dir = args.trace_log_dir
        elif args.per_video_log_dir:
            run_log_root = os.path.dirname(os.path.normpath(args.per_video_log_dir))
            trace_log_dir = os.path.join(run_log_root, "traces")
        else:
            trace_log_dir = config.TRACE_LOG_DIR
        os.makedirs(trace_log_dir, exist_ok=True)
        config.TRACE_LOG_DIR = trace_log_dir
        print(f"[TRACE] Trace logs directory: {config.TRACE_LOG_DIR}")
    _log_model_config()

    with open(args.benchmark_metadata, "r") as f:
        lines = f.readlines()

    results = {}
    for line in lines:
        # one line for one video instance containing multiple questions
        video_info = json.loads(line)
        video_id = video_info["key"]
        qa_list = video_info["qa"]

        qids = [qa["uid"] for qa in qa_list]
        questions = [qa["question"] for qa in qa_list]
        expected_answers = [qa.get("answer") for qa in qa_list]
        question_contexts = [
            {
                "question_type": list(qa.get("question_type", [])),
                "time_reference": str(qa.get("time_reference", "")),
                "video_id": video_id,
                "qid": str(qa.get("uid", "")),
                "video_type": str(video_info.get("type", "")),
            }
            for qa in qa_list
        ]
        trace_ids = [f"{video_id}__{qid}" for qid in qids]

        frames_dir = os.path.join(benchmark_database_folder, video_id, "frames")
        if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
            print(f"Frames for video {frames_dir} not found, skipping...")
            continue
        video_db_path = os.path.join(benchmark_database_folder, video_id, "database.json")
        caption_candidates = [
            os.path.join(caption_root, video_id, "captions", "captions.stage1_eval.json"),
            os.path.join(caption_root, video_id, "captions", "captions.json"),
            os.path.join(benchmark_database_folder, video_id, "captions", "captions.stage1_eval.json"),
            os.path.join(benchmark_database_folder, video_id, "captions", "captions.json"),
        ]
        video_caption_path = None
        for candidate in caption_candidates:
            if os.path.exists(candidate):
                video_caption_path = candidate
                break

        print(f"\n=== VIDEO {video_id} ({len(qa_list)} questions) ===")

        per_video_log = None
        log_handler = None
        orig_stdout, orig_stderr = sys.stdout, sys.stderr
        try:
            if args.per_video_log_dir:
                os.makedirs(args.per_video_log_dir, exist_ok=True)
                per_video_log_path = os.path.join(args.per_video_log_dir, f"{video_id}.log")
                per_video_log = open(per_video_log_path, "w", encoding="utf-8")

                class TeeIO:
                    def __init__(self, *streams):
                        self.streams = streams

                    def write(self, data):
                        for s in self.streams:
                            s.write(data)
                        return len(data)

                    def flush(self):
                        for s in self.streams:
                            s.flush()

                sys.stdout = TeeIO(orig_stdout, per_video_log)
                sys.stderr = TeeIO(orig_stderr, per_video_log)
                # Mirror logging output into per-video logs (logging may use its own handlers).
                log_handler = logging.StreamHandler(per_video_log)
                log_handler.setLevel(logging.INFO)
                log_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
                logging.getLogger().addHandler(log_handler)

            print(f"Initializing SGDCoreAgent from database {video_db_path}...")
            agent = SGDCoreAgent(
                video_db_path,
                video_caption_path=video_caption_path,
                max_iterations=int(getattr(config, "MAX_ITERATIONS", 15)),
                video_file_root=os.path.join(benchmark_database_folder, video_id),
            )
            agent.messages[-1]['content'] += "\nSelect the best option that accurately addresses the question.\nAnswer with the option\'s letter from the given choices directly and only give the best option."
            print("Agent initialized.")
            # Run with questions
            msgs = agent.parallel_run(
                questions,
                max_workers=args.max_workers,
                trace_ids=trace_ids,
                question_contexts=question_contexts,
            )
            for qid, question, expected_answer, trace_id, msg in zip(qids, questions, expected_answers, trace_ids, msgs):
                if not msg:
                    print(f"[WARN] No result for qid={qid}; skipping. See above for tool/API errors.")
                    answer = None
                    answer_check_event = None
                    if config.TRACE_ENABLED:
                        trace_file_path = os.path.join(config.TRACE_LOG_DIR, f"{trace_id}.jsonl")
                        answer_check_event = _append_answer_check_event(
                            trace_file_path=trace_file_path,
                            trace_id=trace_id,
                            predicted_answer=answer,
                            expected_answer=expected_answer,
                        )
                    results[qid] = {
                        "question": question,
                        "answer": answer,
                        "reasoning": None,
                        "expected_answer": expected_answer,
                        "is_correct": answer_check_event.get("is_correct") if answer_check_event else None,
                    }
                    continue
                answer = _extract_answer_from_messages(msg, question)
                answer_check_event = None
                if config.TRACE_ENABLED:
                    trace_file_path = os.path.join(config.TRACE_LOG_DIR, f"{trace_id}.jsonl")
                    answer_check_event = _append_answer_check_event(
                        trace_file_path=trace_file_path,
                        trace_id=trace_id,
                        predicted_answer=answer,
                        expected_answer=expected_answer,
                    )
                results[qid] = {
                    "question": question,
                    "answer": answer,
                    "reasoning": msg,
                    "expected_answer": expected_answer,
                    "is_correct": answer_check_event.get("is_correct") if answer_check_event else None,
                }
        finally:
            if log_handler:
                logging.getLogger().removeHandler(log_handler)
            sys.stdout, sys.stderr = orig_stdout, orig_stderr
            if per_video_log:
                per_video_log.close()

    if args.results_json:
        results_output = args.results_json
    elif args.per_video_log_dir:
        run_log_root = os.path.dirname(os.path.normpath(args.per_video_log_dir))
        results_output = os.path.join(run_log_root, "benchmark_results.json")
    else:
        results_output = os.environ.get("SGD_BENCHMARK_RESULTS_FILE", "benchmark_results.json")

    output_dir = os.path.dirname(os.path.abspath(results_output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(results_output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved benchmark results to: {results_output}")

if __name__ == "__main__":
    main()
