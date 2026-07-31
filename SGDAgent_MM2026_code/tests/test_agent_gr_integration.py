from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import sgd.config as real_config


TOOL_STATE = {
    "gr_result": {},
    "gr_calls": [],
    "sr_result": None,
    "sr_calls": [],
}


class FakeDatabase:
    def get_additional_data(self):
        return {
            "video_length": "00:10:00",
            "stage2_segment_graph_path": "/tmp/nonexistent-stage2.json",
        }


def global_event_query_tool(database, query: str, scope: str | None = None, top_k_events: int = 5) -> str:
    TOOL_STATE["gr_calls"].append(
        {
            "query": query,
            "scope": scope,
            "top_k_events": top_k_events,
        }
    )
    return json.dumps(TOOL_STATE["gr_result"])


def segment_graph_query_tool(
    database,
    query: str,
    time_reference: str | None = None,
    top_k_segments: int = 6,
    max_triples: int = 24,
) -> str:
    TOOL_STATE["sr_calls"].append(
        {
            "query": query,
            "time_reference": time_reference,
            "top_k_segments": top_k_segments,
            "max_triples": max_triples,
        }
    )
    result = TOOL_STATE["sr_result"] or {
        "matched_segments": [
            {
                "segment_id": "seg_0002",
                "t_start": 100.0,
                "t_end": 130.0,
                "score": 1.2,
                "lexical_score": 0.8,
                "matched_relations": [{"predicate": "enters"}],
                "matched_entities": [
                    {"canonical_name": "red car", "type": "vehicle"}
                ],
            }
        ],
        "controller_signals": {
            "localize_only": True,
            "weak_match": False,
        },
    }
    return json.dumps(result)


def frame_graph_query_tool(database, query: str, time_reference: str | None = None) -> str:
    return "{}"


def frame_inspect_tool(database, question: str, time_ranges_hhmmss: list[tuple]) -> str:
    return "visual result"


def clip_search_tool(database, event_description: str, top_k: int = 16) -> str:
    return "caption result"


def init_single_video_db(*args, **kwargs):
    return FakeDatabase()


def _load_sgd_core_module():
    build_database_stub = types.ModuleType("sgd.build_database")
    for func in (
        global_event_query_tool,
        segment_graph_query_tool,
        frame_graph_query_tool,
        frame_inspect_tool,
        clip_search_tool,
        init_single_video_db,
    ):
        setattr(build_database_stub, func.__name__, func)

    schema_stub = types.ModuleType("sgd.func_call_shema")
    schema_stub.as_json_schema = lambda func: {
        "name": func.__name__,
        "parameters": {"type": "object", "properties": {}},
    }
    schema_stub.doc = lambda description: None

    scene_graph_stub = types.ModuleType("sgd.scene_graph")
    scene_graph_stub.__path__ = []
    gei_stub = types.ModuleType("sgd.scene_graph.global_event_index")
    gei_stub.build_global_event_index_from_path = lambda path: {
        "prompt_text": "SHOULD_NOT_BE_INJECTED"
    }

    utils_stub = types.ModuleType("sgd.utils")
    utils_stub.call_openai_model_with_tools = lambda *args, **kwargs: None

    replacements = {
        "sgd.build_database": build_database_stub,
        "sgd.func_call_shema": schema_stub,
        "sgd.scene_graph": scene_graph_stub,
        "sgd.scene_graph.global_event_index": gei_stub,
        "sgd.utils": utils_stub,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        spec = importlib.util.spec_from_file_location(
            "_sgd_core_gr_test",
            PACKAGE_ROOT / "sgd" / "sgd_core.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


CORE = _load_sgd_core_module()


class AgentGRIntegrationTests(unittest.TestCase):
    def setUp(self):
        TOOL_STATE["gr_calls"].clear()
        TOOL_STATE["sr_result"] = None
        TOOL_STATE["sr_calls"].clear()
        TOOL_STATE["gr_result"] = {
            "matched_events": [
                {
                    "event_id": "gei_evt_0002",
                    "t_start": 100.0,
                    "t_end": 130.0,
                }
            ],
            "controller_signals": {
                "weak_match": True,
                "recommended_scope": "00:01:40-00:02:10",
            },
        }
        real_config.GEI_PROMPT_INJECTION_ENABLED = False
        self.agent = CORE.SGDCoreAgent(
            "unused-database.json",
            "unused-captions.json",
            max_iterations=20,
        )

    def _tool_call(self, name: str, arguments: dict, call_id: str):
        return {
            "id": call_id,
            "function": {
                "name": name,
                "arguments": json.dumps(arguments),
            },
        }

    def test_tool_list_exposes_gr_but_not_caption_global_browse(self):
        names = [tool.__name__ for tool in self.agent.tools]
        self.assertEqual(names[0], "global_event_query_tool")
        self.assertIn("segment_graph_query_tool", names)
        self.assertNotIn("global_browse_tool", names)

    def test_default_prompt_does_not_inject_full_timeline(self):
        prompt = self.agent.messages[-1]["content"]
        self.assertNotIn("SHOULD_NOT_BE_INJECTED", prompt)

    def test_count_question_does_not_force_repeated_frame_inspection(self):
        policy = self.agent._build_question_policy(
            "How many people are standing by the car?",
            None,
        )
        self.assertTrue(policy.requires_targeted_fi)
        self.assertFalse(policy.requires_repeated_targeted_fi)

    def test_weak_gr_scope_is_injected_into_next_segment_query(self):
        policy = self.agent._build_question_policy(
            "What happens after the person enters the car?",
            None,
        )
        state = CORE.PolicyState(
            graph_backoff_active=True,
            graph_backoff_reason="previous weak graph result",
        )
        messages = []

        self.agent._exec_tool(
            self._tool_call(
                "global_event_query_tool",
                {"query": "person enters car"},
                "gr-1",
            ),
            messages,
            question="What happens after the person enters the car?",
            question_policy=policy,
            policy_state=state,
        )
        self.assertFalse(state.graph_backoff_active)
        self.assertTrue(state.gr_requires_sr)

        can_finish, _ = self.agent._finish_policy_status("", policy, state)
        self.assertFalse(can_finish)

        self.agent._exec_tool(
            self._tool_call(
                "segment_graph_query_tool",
                {
                    "query": "person enters car",
                    "time_reference": "00:09:00-00:09:30",
                },
                "sr-1",
            ),
            messages,
            question="What happens after the person enters the car?",
            question_policy=policy,
            policy_state=state,
        )
        self.assertEqual(
            TOOL_STATE["sr_calls"][-1]["time_reference"],
            "00:01:40-00:02:10",
        )
        self.assertFalse(state.gr_requires_sr)

    def test_explicit_question_time_takes_precedence_over_gr_scope(self):
        policy = self.agent._build_question_policy(
            "What happens from 00:03:00 to 00:03:20?",
            None,
        )
        state = CORE.PolicyState(
            gr_recommended_scope="00:01:40-00:02:10",
            gr_requires_sr=True,
        )
        messages = []

        self.agent._exec_tool(
            self._tool_call(
                "segment_graph_query_tool",
                {
                    "query": "what happens",
                    "time_reference": "00:09:00-00:09:20",
                },
                "sr-2",
            ),
            messages,
            question="What happens from 00:03:00 to 00:03:20?",
            question_policy=policy,
            policy_state=state,
        )
        self.assertEqual(
            TOOL_STATE["sr_calls"][-1]["time_reference"],
            "00:03:00-00:03:20",
        )

    def test_explicit_question_range_is_applied_as_gr_scope(self):
        policy = self.agent._build_question_policy(
            "What happens from 00:03:00 to 00:03:20?",
            None,
        )
        state = CORE.PolicyState()

        self.agent._exec_tool(
            self._tool_call(
                "global_event_query_tool",
                {"query": "what happens"},
                "gr-timed",
            ),
            [],
            question="What happens from 00:03:00 to 00:03:20?",
            question_policy=policy,
            policy_state=state,
        )
        self.assertEqual(
            TOOL_STATE["gr_calls"][-1]["scope"],
            "00:03:00-00:03:20",
        )

    def test_empty_gr_requires_caption_backoff_before_finish(self):
        TOOL_STATE["gr_result"] = {
            "matched_events": [],
            "controller_signals": {
                "weak_match": True,
                "recommended_scope": None,
            },
        }
        policy = self.agent._build_question_policy("What happens in the video?", None)
        state = CORE.PolicyState()
        messages = []

        self.agent._exec_tool(
            self._tool_call(
                "global_event_query_tool",
                {"query": "what happens"},
                "gr-empty",
            ),
            messages,
            question="What happens in the video?",
            question_policy=policy,
            policy_state=state,
        )
        can_finish, _ = self.agent._finish_policy_status("", policy, state)
        self.assertFalse(can_finish)
        self.assertTrue(state.gr_requires_caption_backoff)

        self.agent._exec_tool(
            self._tool_call(
                "frame_inspect_tool",
                {
                    "question": "what happens",
                    "time_ranges_hhmmss": [["00:00:00", "00:00:10"]],
                },
                "fi-1",
            ),
            messages,
            question="What happens in the video?",
            question_policy=policy,
            policy_state=state,
        )
        self.assertTrue(state.gr_requires_caption_backoff)

        self.agent._exec_tool(
            self._tool_call(
                "clip_search_tool",
                {"event_description": "what happens"},
                "cb-1",
            ),
            messages,
            question="What happens in the video?",
            question_policy=policy,
            policy_state=state,
        )
        self.assertFalse(state.gr_requires_caption_backoff)

    def test_out_of_scope_segment_result_requires_caption_backoff(self):
        policy = self.agent._build_question_policy(
            "What happens after the person enters the car?",
            None,
        )
        state = CORE.PolicyState()
        messages = []
        self.agent._exec_tool(
            self._tool_call(
                "global_event_query_tool",
                {"query": "person enters car"},
                "gr-2",
            ),
            messages,
            question="What happens after the person enters the car?",
            question_policy=policy,
            policy_state=state,
        )
        TOOL_STATE["sr_result"] = {
            "matched_segments": [
                {
                    "segment_id": "seg_0099",
                    "t_start": 500.0,
                    "t_end": 530.0,
                    "score": 1.0,
                    "lexical_score": 0.8,
                    "matched_relations": [{"predicate": "enters"}],
                    "matched_entities": [
                        {"canonical_name": "red car", "type": "vehicle"}
                    ],
                }
            ],
            "controller_signals": {},
        }
        self.agent._exec_tool(
            self._tool_call(
                "segment_graph_query_tool",
                {"query": "person enters car"},
                "sr-outside",
            ),
            messages,
            question="What happens after the person enters the car?",
            question_policy=policy,
            policy_state=state,
        )

        self.assertFalse(state.gr_requires_sr)
        self.assertTrue(state.gr_requires_caption_backoff)
        can_finish, _ = self.agent._finish_policy_status("", policy, state)
        self.assertFalse(can_finish)

    def test_max_iteration_fallback_cannot_bypass_gr_followup(self):
        self.agent.max_iterations = 1
        responses = iter(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        self._tool_call(
                            "global_event_query_tool",
                            {"query": "person enters car"},
                            "gr-run",
                        )
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        self._tool_call(
                            "finish",
                            {"answer": "A"},
                            "finish-run",
                        )
                    ],
                },
                {"role": "assistant", "content": "A", "tool_calls": []},
            ]
        )
        call_count = 0

        def fake_orchestrator(_agent, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return next(responses)

        self.agent._call_orchestrator_with_retries = types.MethodType(
            fake_orchestrator,
            self.agent,
        )
        question = (
            "What happens next? "
            "(A) The car moves. (B) The car stops. "
            "(C) The person leaves. (D) Nothing changes."
        )

        result = self.agent.run(question)

        self.assertIsNone(result)
        self.assertEqual(call_count, 2)


if __name__ == "__main__":
    unittest.main()
