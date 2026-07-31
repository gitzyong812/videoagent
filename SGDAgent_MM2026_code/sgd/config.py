import os

# ------------------ video download and segmentation configuration ------------------ #
VIDEO_DATABASE_FOLDER = "./video_database/"
VIDEO_FPS = 2 # frames per second
CLIP_SECS = 10 # seconds

# ------------------ model configuration ------------------ #
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None) # will overwrite Azure OpenAI setting
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

AOAI_CAPTION_VLM_ENDPOINT_LIST = [""]
AOAI_CAPTION_VLM_MODEL_NAME = "gpt-4.1-mini"

AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST = [""]
AOAI_ORCHESTRATOR_LLM_MODEL_NAME = "o3"

AOAI_TOOL_VLM_ENDPOINT_LIST = [""]
AOAI_TOOL_VLM_MODEL_NAME = "gpt-4.1-mini"
AOAI_TOOL_VLM_MAX_FRAME_NUM = 50

AOAI_EMBEDDING_RESOURCE_LIST = [""]
AOAI_EMBEDDING_LARGE_MODEL_NAME = "text-embedding-3-large"
AOAI_EMBEDDING_LARGE_DIM = 3072

# ------------------ agent and tool setting ------------------ #
GLOBAL_BROWSE_TOPK = 300
OVERWRITE_CLIP_SEARCH_TOPK = 0 # 0 means no overwrite and let agent decide
SCENE_GRAPH_TOPK_FRAMES = 8
SCENE_GRAPH_TOPK_SEGMENTS = 6
SCENE_GRAPH_MAX_TRIPLES = 24

# ------------------ runtime policy gates ------------------ #
POLICY_TARGETED_FI_MAX_WINDOW_SEC = 18.0
POLICY_TARGETED_FI_MIN_WINDOW_SEC = 6.0
POLICY_TARGETED_FI_WINDOW_MULTIPLIER = 0.6
POLICY_REPEAT_TARGETED_FI_FOR_FINE_DETAIL = False
POLICY_STRICT_GRAPH_TIME_REFERENCE = False
POLICY_GRAPH_MIN_LEXICAL_SCORE = 0.15
POLICY_GRAPH_MIN_TOP_MARGIN = 0.12
POLICY_STAGNATION_WINDOW = 3
POLICY_FI_OUTPUT_SIMILARITY_THRESHOLD = 0.55
POLICY_REPEATED_FI_MIN_OVERLAP_RATIO = 0.5
POLICY_MAX_FINISH_REJECTIONS = 3

SINGLE_CHOICE_QA = True  # Design for benchmark test. If True, the agent will only return options for single-choice questions.
MAX_ITERATIONS = 20  # Used by benchmark/VQA runs; 20 helps a few hard multi-step cases, but finalization logic remains primary.
ORCHESTRATOR_RETRY_ATTEMPTS = 5
ORCHESTRATOR_RETRY_BASE_DELAY_SEC = 1.0
PARALLEL_RUN_RETRY_ATTEMPTS = 2
TOOL_CALL_RETRY_ATTEMPTS = 3
TOOL_CALL_RETRY_BASE_DELAY_SEC = 2.0

# ------------------ stage1 scene graph (frame-level) ------------------ #
SCENE_GRAPH_STAGE1_ENABLED = True
SCENE_GRAPH_STAGE1_USE_VLM = True
SCENE_GRAPH_STAGE1_VLM_ENDPOINT_LIST = AOAI_TOOL_VLM_ENDPOINT_LIST
SCENE_GRAPH_STAGE1_VLM_MODEL_NAME = "gpt-4.1"
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
# ---- stage12-boost-v1: ablation flags (entity alignment) ---- #
SCENE_GRAPH_STAGE1_USE_SYNONYM_MAP = True           # synonym bonus in entity matching
SCENE_GRAPH_STAGE1_POST_MERGE = True                # post-build entity merge pass
SCENE_GRAPH_STAGE1_POST_MERGE_THRESHOLD = 0.70      # lower threshold for post-merge
SCENE_GRAPH_STAGE1_USE_EXTENDED_PREDICATES = True    # enable extended predicate vocabulary
SCENE_GRAPH_STAGE1_EXTENDED_PREDICATES = [
    # communicative
    "talking_to", "listening_to", "interacting_with",
    # functional
    "driving", "riding", "eating", "reading",
    "watching", "carrying", "playing_with",
    # causal / temporal
    "caused_by", "leads_to", "follows", "part_of",
]
SCENE_GRAPH_STAGE1_ENHANCED_FALLBACK = True         # SVO extraction from captions
SCENE_GRAPH_STAGE1_NEIGHBOR_PROPAGATION = True       # propagate entities from neighbor frame

# ------------------ stage2 scene graph (segment-level) ------------------ #
SCENE_GRAPH_STAGE2_ENABLED = True
SCENE_GRAPH_STAGE2_FORCE_REBUILD = False
SCENE_GRAPH_STAGE2_FILE_NAME = "scene_graph_stage2_segment.json"
SCENE_GRAPH_STAGE2_WINDOW_SECS = 30
SCENE_GRAPH_STAGE2_OVERLAP_SECS = 5
SCENE_GRAPH_STAGE2_MAX_RELATIONS_PER_SEGMENT = 64
SCENE_GRAPH_STAGE2_EMBED_SEGMENTS = True  # embed segment text for semantic retrieval fallback
SCENE_GRAPH_STAGE2_EMBED_BATCH_SIZE = 64  # segments per embedding API call
# ---- stage12-boost-v1: segment enrichment flags ---- #
SCENE_GRAPH_STAGE2_INCLUDE_CAPTION_SNIPPETS = True   # carry caption text into segments
SCENE_GRAPH_STAGE2_MAX_CAPTION_SNIPPETS = 4          # max caption snippets per segment

# ------------------ global event index (prompt-injected, no extra tool) ---- #
GEI_ENABLED = True  # inject video timeline into prompt from Stage 2 graph
GEI_MAX_EVENTS = 12  # target scene count; aggressive compression from ~120 segments
GEI_MAX_CHARS = 2400  # hard cap on the text summary length
GEI_MERGE_GAP_SEC = 5.0  # merge adjacent segments sharing entities within this gap
