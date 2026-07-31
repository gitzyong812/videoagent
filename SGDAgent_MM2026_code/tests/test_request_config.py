from __future__ import annotations

import importlib
import os
import sys
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

    if "azure.identity" not in sys.modules:
        azure = sys.modules.setdefault("azure", types.ModuleType("azure"))
        identity = types.ModuleType("azure.identity")

        class AzureCliCredential:
            def get_token(self, _scope):
                return types.SimpleNamespace(token="test-token")

        identity.AzureCliCredential = AzureCliCredential
        azure.identity = identity
        sys.modules["azure.identity"] = identity


_install_import_stubs()

import sgd.config as config
import sgd.utils as utils


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        raise AssertionError("raise_for_status should not be called")


class RequestConfigTests(unittest.TestCase):
    def test_default_models_and_environment_overrides(self):
        keys = {
            "SGD_ORCHESTRATOR_MODEL",
            "SGD_TOOL_VLM_MODEL",
            "SGD_STAGE1_VLM_MODEL",
            "SGD_CAPTION_VLM_MODEL",
            "SGD_CAPTION_WORKERS",
            "SGD_TOOL_VLM_ENDPOINTS",
        }
        original = {key: os.environ.get(key) for key in keys}
        try:
            with mock.patch.dict(os.environ, {}, clear=True):
                importlib.reload(config)
                self.assertEqual(config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME, "o4-mini")
                self.assertEqual(config.AOAI_TOOL_VLM_MODEL_NAME, "o3")
                self.assertEqual(config.SCENE_GRAPH_STAGE1_VLM_MODEL_NAME, "gpt-4.1")
                self.assertEqual(config.CAPTION_WORKERS, 16)

            with mock.patch.dict(
                os.environ,
                {
                    "SGD_ORCHESTRATOR_MODEL": "test-reasoner",
                    "SGD_TOOL_VLM_MODEL": "test-fi",
                    "SGD_CAPTION_WORKERS": "0",
                    "SGD_TOOL_VLM_ENDPOINTS": " https://one.example/ ,https://two.example ",
                },
                clear=True,
            ):
                importlib.reload(config)
                self.assertEqual(config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME, "test-reasoner")
                self.assertEqual(config.AOAI_TOOL_VLM_MODEL_NAME, "test-fi")
                self.assertEqual(config.CAPTION_WORKERS, 1)
                self.assertEqual(
                    config.AOAI_TOOL_VLM_ENDPOINT_LIST,
                    ["https://one.example", "https://two.example"],
                )
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            importlib.reload(config)

    def test_nonpositive_video_settings_are_rejected(self):
        original_fps = os.environ.get("SGD_VIDEO_FPS")
        original_clip = os.environ.get("SGD_CLIP_SECS")
        try:
            with mock.patch.dict(
                os.environ,
                {"SGD_VIDEO_FPS": "0"},
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "SGD_VIDEO_FPS"):
                    importlib.reload(config)
            with mock.patch.dict(
                os.environ,
                {"SGD_CLIP_SECS": "0"},
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "SGD_CLIP_SECS"):
                    importlib.reload(config)
        finally:
            if original_fps is None:
                os.environ.pop("SGD_VIDEO_FPS", None)
            else:
                os.environ["SGD_VIDEO_FPS"] = original_fps
            if original_clip is None:
                os.environ.pop("SGD_CLIP_SECS", None)
            else:
                os.environ["SGD_CLIP_SECS"] = original_clip
            importlib.reload(config)

    def test_standard_model_payload_has_temperature_and_completion_token_limit(self):
        response = FakeResponse(
            {"model": "gpt-4.1", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )
        with mock.patch.object(utils.requests, "post", return_value=response) as post:
            result = utils.call_openai_model_with_tools(
                [{"role": "user", "content": "hello"}],
                endpoints=[""],
                model_name="gpt-4.1",
                api_key="test-key",
                max_tokens=800,
                temperature=0.25,
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["max_completion_tokens"], 800)
        self.assertEqual(payload["temperature"], 0.25)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(post.call_args.kwargs["timeout"], config.API_TIMEOUT_SECS)
        self.assertEqual(result["content"], "ok")

    def test_reasoning_model_payload_uses_completion_token_limit(self):
        response = FakeResponse(
            {"model": "o4-mini", "choices": [{"message": {"role": "assistant", "content": "A"}}]}
        )
        with mock.patch.object(utils.requests, "post", return_value=response) as post:
            utils.call_openai_model_with_tools(
                [{"role": "user", "content": "hello"}],
                endpoints=[""],
                model_name="o4-mini",
                api_key="test-key",
                max_tokens=512,
                temperature=0.0,
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["max_completion_tokens"], 512)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("temperature", payload)

    def test_reasoning_model_detection_accepts_other_o_series_names(self):
        self.assertTrue(utils._uses_reasoning_token_parameter("o5-mini"))
        self.assertFalse(utils._uses_reasoning_token_parameter("gpt-4.1"))

    def test_embedding_request_uses_json_and_timeout(self):
        response = FakeResponse({"data": [{"embedding": [0.1, 0.2]}]})
        with mock.patch.object(utils.requests, "post", return_value=response) as post:
            data = utils.AzureOpenAIEmbeddingService.get_embeddings(
                endpoints=[""],
                model_name="text-embedding-3-large",
                input_text=["hello"],
                api_key="test-key",
            )

        self.assertEqual(data, [{"embedding": [0.1, 0.2]}])
        self.assertEqual(post.call_args.kwargs["json"]["input"], ["hello"])
        self.assertEqual(post.call_args.kwargs["timeout"], config.API_TIMEOUT_SECS)

    def test_empty_azure_endpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No Azure endpoint configured"):
            utils.call_openai_model_with_tools(
                [{"role": "user", "content": "hello"}],
                endpoints=[""],
                model_name="o4-mini",
                api_key=None,
            )

    def test_retry_exhaustion_preserves_the_error(self):
        call_count = 0

        def always_times_out():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("request timed out")

        wrapped = utils.retry_with_exponential_backoff(
            always_times_out,
            initial_delay=0,
            exponential_base=1,
            jitter=False,
            max_retries=1,
        )
        with mock.patch.object(utils.time, "sleep"):
            with self.assertRaisesRegex(TimeoutError, "request timed out"):
                wrapped()
        self.assertEqual(call_count, 2)


if __name__ == "__main__":
    unittest.main()
