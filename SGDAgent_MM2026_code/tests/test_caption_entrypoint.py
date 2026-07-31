from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _install_import_stubs() -> None:
    if "cv2" not in sys.modules:
        cv2 = types.ModuleType("cv2")
        cv2.imread = lambda _path: object()
        sys.modules["cv2"] = cv2

    if "requests" not in sys.modules:
        requests = types.ModuleType("requests")
        requests.post = None
        sys.modules["requests"] = requests

    if "nano_vectordb" not in sys.modules:
        nano_vectordb = types.ModuleType("nano_vectordb")

        class NanoVectorDB:
            pass

        nano_vectordb.NanoVectorDB = NanoVectorDB
        sys.modules["nano_vectordb"] = nano_vectordb

    if "azure.identity" not in sys.modules:
        azure = sys.modules.setdefault("azure", types.ModuleType("azure"))
        identity = types.ModuleType("azure.identity")

        class AzureCliCredential:
            pass

        identity.AzureCliCredential = AzureCliCredential
        azure.identity = identity
        sys.modules["azure.identity"] = identity


_install_import_stubs()

import local_run
import sgd.frame_caption as frame_caption


class FakePool:
    processes = None

    def __init__(self, processes):
        type(self).processes = processes

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False

    def imap_unordered(self, func, items):
        return [func(item) for item in items]


class CaptionEntrypointTests(unittest.TestCase):
    def test_caption_clip_retries_none_without_indexing_error(self):
        task = ("0_9", {"files": [], "transcript": "No transcript."})
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                frame_caption,
                "call_openai_model_with_tools",
                return_value=None,
            ) as call:
                timestamp, payload = frame_caption._caption_clip(task, tmpdir)

        self.assertEqual(timestamp, "0_9")
        self.assertEqual(payload, {})
        self.assertEqual(call.call_count, 3)

    def test_caption_clip_validates_and_normalizes_payload(self):
        task = ("0_9", {"files": [], "transcript": "A person enters."})
        response = {
            "content": json.dumps(
                {
                    "clip_description": "A person enters the room.",
                    "subject_registry": {"subject_1": {"name": "person"}},
                }
            )
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                frame_caption,
                "call_openai_model_with_tools",
                return_value=response,
            ):
                timestamp, payload = frame_caption._caption_clip(task, tmpdir)

            checkpoint = Path(tmpdir) / "0_9.json"
            self.assertTrue(checkpoint.exists())

        self.assertEqual(timestamp, "0_9")
        self.assertEqual(payload["clip_start_time"], "00:00:00")
        self.assertEqual(payload["clip_end_time"], "00:00:09")
        self.assertIn("A person enters.", payload["clip_description"])

    def test_process_video_clamps_worker_count_to_one(self):
        clip = ("0_9", {"files": [], "transcript": "No transcript."})
        parsed = {
            "clip_description": "A person enters.",
            "subject_registry": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.object(frame_caption, "gather_clip_frames", return_value=[clip]),
                mock.patch.object(
                    frame_caption,
                    "_caption_clip",
                    return_value=("0_9", parsed),
                ),
                mock.patch.object(
                    frame_caption,
                    "merge_subject_registries",
                    return_value={},
                ),
                mock.patch.object(frame_caption.mp, "Pool", FakePool),
                mock.patch.object(frame_caption.config, "CAPTION_WORKERS", 0),
            ):
                frame_caption.process_video("frames", tmpdir)

            with open(Path(tmpdir) / "captions.json", encoding="utf-8") as f:
                captions = json.load(f)

        self.assertEqual(FakePool.processes, 1)
        self.assertEqual(captions["0_9"]["caption"], "A person enters.")

    def test_process_video_rejects_empty_frame_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(frame_caption, "gather_clip_frames", return_value=[]):
                with self.assertRaisesRegex(ValueError, "No decoded video frames"):
                    frame_caption.process_video("frames", tmpdir)

    def test_process_video_does_not_cache_partial_caption_results(self):
        clips = [
            ("0_9", {"files": [], "transcript": "No transcript."}),
            ("10_19", {"files": [], "transcript": "No transcript."}),
        ]
        valid = {
            "clip_description": "A person enters.",
            "subject_registry": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.object(frame_caption, "gather_clip_frames", return_value=clips),
                mock.patch.object(
                    frame_caption,
                    "_caption_clip",
                    side_effect=[("0_9", valid), ("10_19", {})],
                ),
                mock.patch.object(frame_caption.mp, "Pool", FakePool),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Caption generation failed for 1 clip",
                ):
                    frame_caption.process_video("frames", tmpdir)

            self.assertFalse((Path(tmpdir) / "captions.json").exists())

    def test_answer_from_messages_accepts_only_terminal_answers(self):
        self.assertIsNone(local_run._answer_from_messages(None))
        self.assertIsNone(local_run._answer_from_messages([]))
        rejected_intermediate = [
            {"role": "assistant", "content": "answer"},
            {"role": "tool", "content": "observation"},
        ]
        self.assertIsNone(local_run._answer_from_messages(rejected_intermediate))
        direct_terminal = [
            {"role": "tool", "content": "observation"},
            {"role": "assistant", "content": "answer", "tool_calls": []},
        ]
        self.assertEqual(
            local_run._answer_from_messages(direct_terminal),
            "answer",
        )
        finish_terminal = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "finish",
                            "arguments": '{"answer": "B"}',
                        }
                    }
                ],
            }
        ]
        self.assertEqual(
            local_run._answer_from_messages(finish_terminal),
            "B",
        )


if __name__ == "__main__":
    unittest.main()
