import os


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_endpoint_list(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    endpoints = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    return endpoints or [""]


# ------------------ video download and segmentation configuration ------------------ #
VIDEO_DATABASE_FOLDER = os.environ.get("SGD_VIDEO_DATABASE_FOLDER", "./video_database/")
VIDEO_FPS = _env_float("SGD_VIDEO_FPS", 2.0)  # frames per second
CLIP_SECS = _env_int("SGD_CLIP_SECS", 10)  # seconds
if VIDEO_FPS <= 0:
    raise ValueError("SGD_VIDEO_FPS must be greater than zero.")
if CLIP_SECS <= 0:
    raise ValueError("SGD_CLIP_SECS must be greater than zero.")
CAPTION_WORKERS = max(1, _env_int("SGD_CAPTION_WORKERS", 16))
API_TIMEOUT_SECS = max(1.0, _env_float("SGD_API_TIMEOUT_SECS", 600.0))

# ------------------ model configuration ------------------ #
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None) # will overwrite Azure OpenAI setting
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

AOAI_CAPTION_VLM_ENDPOINT_LIST = _env_endpoint_list("SGD_CAPTION_VLM_ENDPOINTS")
AOAI_CAPTION_VLM_MODEL_NAME = os.environ.get("SGD_CAPTION_VLM_MODEL", "gpt-4.1-mini")

AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST = _env_endpoint_list("SGD_ORCHESTRATOR_ENDPOINTS")
AOAI_ORCHESTRATOR_LLM_MODEL_NAME = os.environ.get("SGD_ORCHESTRATOR_MODEL", "o4-mini")

AOAI_TOOL_VLM_ENDPOINT_LIST = _env_endpoint_list("SGD_TOOL_VLM_ENDPOINTS")
AOAI_TOOL_VLM_MODEL_NAME = os.environ.get("SGD_TOOL_VLM_MODEL", "o3")
# Frame inspection uniformly samples at most this many frames over all requested ranges.
AOAI_TOOL_VLM_MAX_FRAME_NUM = max(1, _env_int("SGD_TOOL_VLM_MAX_FRAMES", 50))

AOAI_EMBEDDING_RESOURCE_LIST = _env_endpoint_list("SGD_EMBEDDING_ENDPOINTS")
AOAI_EMBEDDING_LARGE_MODEL_NAME = os.environ.get(
    "SGD_EMBEDDING_MODEL", "text-embedding-3-large"
)
AOAI_EMBEDDING_LARGE_DIM = _env_int("SGD_EMBEDDING_DIM", 3072)

# ------------------ agent and tool setting ------------------ #
GLOBAL_BROWSE_TOPK = 300
OVERWRITE_CLIP_SEARCH_TOPK = 0 # 0 means no overwrite and let agent decide
GLOBAL_EVENT_TOPK = 5
SCENE_GRAPH_TOPK_FRAMES = 8
SCENE_GRAPH_TOPK_SEGMENTS = 6
SCENE_GRAPH_MAX_TRIPLES = 24

# ------------------ runtime policy gates ------------------ #
POLICY_TARGETED_FI_MAX_WINDOW_SEC = 18.0
POLICY_TARGETED_FI_MIN_WINDOW_SEC = 6.0
POLICY_TARGETED_FI_WINDOW_MULTIPLIER = 0.6
POLICY_REPEAT_TARGETED_FI_FOR_FINE_DETAIL = False
POLICY_STRICT_GRAPH_TIME_REFERENCE = True
POLICY_GRAPH_MIN_LEXICAL_SCORE = 0.15
POLICY_GRAPH_MIN_TOP_MARGIN = 0.12
POLICY_STAGNATION_WINDOW = 3
POLICY_FI_OUTPUT_SIMILARITY_THRESHOLD = 0.55
POLICY_REPEATED_FI_MIN_OVERLAP_RATIO = 0.5
POLICY_MAX_FINISH_REJECTIONS = 3

MAX_ITERATIONS = max(1, _env_int("SGD_MAX_ITERATIONS", 20))
ORCHESTRATOR_RETRY_ATTEMPTS = 5
ORCHESTRATOR_RETRY_BASE_DELAY_SEC = 1.0
PARALLEL_RUN_RETRY_ATTEMPTS = 2
TOOL_CALL_RETRY_ATTEMPTS = 3
TOOL_CALL_RETRY_BASE_DELAY_SEC = 2.0

# ------------------ stage1 scene graph (frame-level) ------------------ #
SCENE_GRAPH_STAGE1_ENABLED = True
SCENE_GRAPH_STAGE1_USE_VLM = True
SCENE_GRAPH_STAGE1_VLM_ENDPOINT_LIST = AOAI_TOOL_VLM_ENDPOINT_LIST
SCENE_GRAPH_STAGE1_VLM_MODEL_NAME = os.environ.get("SGD_STAGE1_VLM_MODEL", "gpt-4.1")
SCENE_GRAPH_STAGE1_FORCE_REBUILD = False
SCENE_GRAPH_STAGE1_FILE_NAME = "scene_graph_stage1.json"
SCENE_GRAPH_STAGE1_SAMPLE_SECS = 5
SCENE_GRAPH_STAGE1_MAX_KEYFRAMES = 120
SCENE_GRAPH_STAGE1_ENTITY_MATCH_THRESHOLD = 0.78
SCENE_GRAPH_STAGE1_CAPTION_FALLBACK = True
SCENE_GRAPH_STAGE1_FALLBACK_MAX_ENTITIES = 4
SCENE_GRAPH_STAGE1_ALLOWED_PREDICATES = [
    "left_of",
    "right_of",
    "behind",
    "in_front_of",
    "holding",
    "wearing",
    "on",
    "under",
    "next_to",
    "facing",
    "looking_at",
    "sitting_on",
    "standing_on",
    "opening",
    "closing",
    "pointing_to",
    "touching",
    "inside",
]

# ------------------ stage2 scene graph (segment-level) ------------------ #
SCENE_GRAPH_STAGE2_ENABLED = True
SCENE_GRAPH_STAGE2_FORCE_REBUILD = False
SCENE_GRAPH_STAGE2_FILE_NAME = "scene_graph_stage2_segment.json"
SCENE_GRAPH_STAGE2_WINDOW_SECS = 30
SCENE_GRAPH_STAGE2_OVERLAP_SECS = 5
SCENE_GRAPH_STAGE2_MAX_RELATIONS_PER_SEGMENT = 64
SCENE_GRAPH_STAGE2_EMBED_SEGMENTS = True  # embed segment text for semantic retrieval fallback
SCENE_GRAPH_STAGE2_EMBED_BATCH_SIZE = 64  # segments per embedding API call

# ------------------ optional prompt-level global event index ------------------ #
GEI_PROMPT_INJECTION_ENABLED = _env_bool(
    "SGD_GEI_PROMPT_INJECTION_ENABLED",
    False,
)
GEI_MAX_EVENTS = 12  # bounded event-region budget
GEI_MAX_CHARS = 2400  # hard cap on the text summary length
GEI_MERGE_GAP_SEC = 5.0  # merge adjacent segments sharing entities within this gap
