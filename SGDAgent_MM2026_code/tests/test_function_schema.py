from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from typing import Annotated


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

from sgd.func_call_shema import as_json_schema, doc


class DummyDatabase:
    pass


def sample_tool(
    database: Annotated[DummyDatabase, doc("Internal database.")],
    query: Annotated[str, doc("Question text.")],
    top_k: Annotated[int, doc("Result count.")] = 3,
) -> str:
    """Sample tool."""
    return query


class FunctionSchemaTests(unittest.TestCase):
    def test_resolves_postponed_annotations_and_hides_database(self):
        schema = as_json_schema(sample_tool)
        parameters = schema["parameters"]

        self.assertEqual(set(parameters["properties"]), {"query", "top_k"})
        self.assertEqual(parameters["properties"]["query"]["description"], "Question text.")
        self.assertEqual(parameters["properties"]["top_k"]["default"], 3)
        self.assertEqual(parameters["required"], ["query"])

    def test_all_agent_tool_schemas_are_buildable(self):
        from sgd.build_database import (
            clip_search_tool,
            frame_graph_query_tool,
            frame_inspect_tool,
            global_event_query_tool,
            global_browse_tool,
            segment_graph_query_tool,
        )
        from sgd.sgd_core import finish

        tools = [
            global_event_query_tool,
            segment_graph_query_tool,
            frame_graph_query_tool,
            frame_inspect_tool,
            clip_search_tool,
            global_browse_tool,
            finish,
        ]
        for tool in tools:
            with self.subTest(tool=tool.__name__):
                schema = as_json_schema(tool)
                self.assertNotIn("database", schema["parameters"].get("properties", {}))

    def test_global_event_tool_has_public_basic_gr_contract(self):
        from sgd.build_database import global_event_query_tool

        schema = as_json_schema(global_event_query_tool)
        parameters = schema["parameters"]
        properties = parameters["properties"]

        self.assertEqual(
            set(properties),
            {"query", "scope", "top_k_events"},
        )
        self.assertEqual(properties["top_k_events"]["default"], 5)
        scope_types = {
            variant.get("type")
            for variant in properties["scope"].get("anyOf", [])
        }
        self.assertEqual(scope_types, {"string", "null"})


if __name__ == "__main__":
    unittest.main()
