import unittest
import sys
import types
import json
import inspect
import tempfile
from pathlib import Path
from unittest.mock import patch


if "cv2" not in sys.modules:
    try:
        __import__("cv2")
    except ModuleNotFoundError:
        sys.modules["cv2"] = types.ModuleType("cv2")
if "nano_vectordb" not in sys.modules:
    try:
        __import__("nano_vectordb")
    except ModuleNotFoundError:
        nano_module = types.ModuleType("nano_vectordb")
        nano_module.NanoVectorDB = type("NanoVectorDB", (), {})
        sys.modules["nano_vectordb"] = nano_module
try:
    __import__("azure.identity")
except ModuleNotFoundError:
    azure_module = types.ModuleType("azure")
    identity_module = types.ModuleType("azure.identity")
    identity_module.AzureCliCredential = type("AzureCliCredential", (), {})
    azure_module.identity = identity_module
    sys.modules["azure"] = azure_module
    sys.modules["azure.identity"] = identity_module

try:
    from sgd.build_database import (
        caption_backoff_tool,
        frame_graph_query_tool,
        global_graph_query_tool,
        segment_graph_query_tool,
    )
    import sgd.build_database as build_database
    import sgd.frame_caption as frame_caption
    import sgd.utils as model_utils
    from sgd.func_call_shema import as_json_schema
    from sgd.sgd_core import PolicyState, QuestionPolicy, SGDCoreAgent
except ModuleNotFoundError:
    PolicyState = QuestionPolicy = SGDCoreAgent = None


@unittest.skipIf(SGDCoreAgent is None, "optional runtime dependencies are not installed")
class AgentPolicyTest(unittest.TestCase):
    def setUp(self):
        self.agent = SGDCoreAgent.__new__(SGDCoreAgent)

    def test_query_type_routing(self):
        self.assertEqual(self.agent._initial_tool_name("Why did the person leave?"), "global_graph_query_tool")
        self.assertEqual(self.agent._initial_tool_name("When does the person leave?"), "segment_graph_query_tool")
        self.assertEqual(self.agent._initial_tool_name("What color is the shirt?"), "frame_inspect_tool")
        self.assertEqual(self.agent._initial_tool_name("Where is the cup?"), "frame_graph_query_tool")

    def test_localization_only_blocks_finish(self):
        policy = QuestionPolicy((), "", "", False, False, False, "detail")
        state = PolicyState(
            last_graph_requires_followup=True,
            last_graph_recommended_action="segment_graph_query_tool",
        )
        allowed, reason = self.agent._finish_policy_status("question", policy, state)
        self.assertFalse(allowed)
        self.assertIn("segment_graph_query_tool", reason)

    def test_paper_tool_schemas_are_exportable(self):
        schemas = [
            as_json_schema(tool)
            for tool in (
                global_graph_query_tool,
                segment_graph_query_tool,
                frame_graph_query_tool,
                caption_backoff_tool,
            )
        ]
        self.assertEqual(
            [schema["name"] for schema in schemas],
            [
                "global_graph_query_tool",
                "segment_graph_query_tool",
                "frame_graph_query_tool",
                "caption_backoff_tool",
            ],
        )
        self.assertEqual(
            inspect.signature(global_graph_query_tool).parameters["top_k_regions"].default,
            5,
        )

    def test_database_metadata_links_all_graph_levels(self):
        class FakeVectorDB:
            def __init__(self, emb_dim, storage_file):
                self.storage_file = storage_file
                self.additional_data = {}

            def upsert(self, data):
                return data

            def get_additional_data(self):
                return self.additional_data

            def store_additional_data(self, **kwargs):
                self.additional_data.update(kwargs)

            def save(self):
                Path(self.storage_file).write_text(json.dumps(self.additional_data))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captions_path = root / "captions.json"
            captions_path.write_text(
                json.dumps({"0_10": {"caption": "a person enters"}, "subject_registry": {}})
            )
            stage1 = str(root / "scene_graph_stage1.json")
            stage2 = str(root / "scene_graph_stage2_segment.json")
            global_graph = str(root / "scene_graph_global.json")
            database_path = str(root / "database.json")
            with patch.object(build_database, "NanoVectorDB", FakeVectorDB), patch.object(
                build_database,
                "preprocess_captions",
                return_value=[((0.0, 10.0), {"caption": "a person enters"}, [0.0, 0.0])],
            ), patch.object(build_database, "_ensure_stage1_graph_file", return_value=stage1), patch.object(
                build_database, "_ensure_stage2_graph_file", return_value=stage2
            ), patch.object(build_database, "_ensure_global_graph_file", return_value=global_graph):
                database = build_database.init_single_video_db(
                    str(captions_path), database_path, 2, video_file_root_override=str(root)
                )
            self.assertTrue(Path(database_path).exists())
            metadata = database.get_additional_data()
            self.assertEqual(metadata["stage1_scene_graph_path"], stage1)
            self.assertEqual(metadata["stage2_segment_graph_path"], stage2)
            self.assertEqual(metadata["global_event_graph_path"], global_graph)

    def test_stage1_build_failure_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            build_database.config, "SCENE_GRAPH_STAGE1_ENABLED", True
        ), patch.object(
            build_database.config, "SCENE_GRAPH_STAGE1_FORCE_REBUILD", True
        ), patch.object(
            build_database, "ensure_stage1_scene_graph", side_effect=RuntimeError("VLM unavailable")
        ):
            captions_path = Path(tmp) / "captions.json"
            captions_path.write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "VLM unavailable"):
                build_database._ensure_stage1_graph_file(str(captions_path))

    def test_model_payload_respects_role_settings(self):
        class FakeResponse:
            status_code = 200
            text = ""

            def json(self):
                return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        payloads = []

        def fake_post(url, headers, json=None, **kwargs):
            payloads.append(json)
            return FakeResponse()

        with patch.object(model_utils.requests, "post", side_effect=fake_post):
            model_utils.call_openai_model_with_tools(
                [{"role": "user", "content": "test"}], [""], "gpt-4.1",
                api_key="test", max_tokens=800, temperature=0.0,
            )
            model_utils.call_openai_model_with_tools(
                [{"role": "user", "content": "test"}], [""], "o4-mini",
                api_key="test", max_tokens=400,
            )
        self.assertEqual(payloads[0]["max_tokens"], 800)
        self.assertEqual(payloads[0]["temperature"], 0.0)
        self.assertEqual(payloads[1]["max_completion_tokens"], 400)
        self.assertNotIn("temperature", payloads[1])

    def test_caption_contract_is_valid_json_and_fails_clearly(self):
        template = frame_caption.CAPTION_PROMPT.split("Output template:\n", 1)[1]
        rendered = template.replace("CLIP_START_TIME", "00:00:00").replace(
            "CLIP_END_TIME", "00:00:09"
        )
        parsed = json.loads(rendered)
        self.assertIn("clip_description", parsed)
        self.assertIsInstance(parsed["subject_registry"], dict)

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            frame_caption, "call_openai_model_with_tools", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "Caption VLM extraction failed"):
                frame_caption._caption_clip(
                    ("0_9", {"files": [], "transcript": "No transcript."}),
                    tmp,
                )


if __name__ == "__main__":
    unittest.main()
