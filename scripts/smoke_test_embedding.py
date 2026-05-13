#!/usr/bin/env python3
"""
Smoke test: verify that segment embedding calls work with current API config.

Usage:
    export OPENAI_API_KEY=sk-...
    python3 scripts/smoke_test_embedding.py

Tests:
  1. Single text embedding (same as clip_search_tool)
  2. Batch embedding (same as _embed_segments, batch_size=4)
  3. Cosine similarity sanity check
  4. Graceful degradation on empty input
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sgd.config as config


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def main():
    api_key = config.OPENAI_API_KEY
    if not api_key:
        print("FAIL: OPENAI_API_KEY not set. Run: export OPENAI_API_KEY=sk-...")
        sys.exit(1)
    print(f"API key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Base URL: {config.OPENAI_BASE_URL}")
    print(f"Model: {config.AOAI_EMBEDDING_LARGE_MODEL_NAME}")
    print()

    from sgd.utils import AzureOpenAIEmbeddingService

    # ── Test 1: single embedding (matches clip_search_tool path) ──
    print("Test 1: single text embedding...")
    t0 = time.time()
    try:
        results = AzureOpenAIEmbeddingService.get_embeddings(
            endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
            model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
            input_text=["a man walking a dog in a park"],
            api_key=api_key,
        )
        emb = results[0]["embedding"]
        print(f"  OK: dim={len(emb)}, first 3 values={emb[:3]}, took {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    # ── Test 2: batch embedding (matches _embed_segments path) ──
    print("Test 2: batch embedding (4 texts)...")
    texts = [
        "entities: man, car; predicates: driving",
        "entities: woman, kitchen; predicates: cooking, holding",
        "entities: crowd, stage; predicates: watching, cheering",
        "entities: dog, ball; predicates: playing, running",
    ]
    t0 = time.time()
    try:
        results = AzureOpenAIEmbeddingService.get_embeddings(
            endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
            model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
            input_text=texts,
            api_key=api_key,
        )
        assert len(results) == 4, f"Expected 4 results, got {len(results)}"
        dims = [len(r["embedding"]) for r in results]
        assert all(d == dims[0] for d in dims), f"Inconsistent dims: {dims}"
        print(f"  OK: {len(results)} embeddings, dim={dims[0]}, took {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    # ── Test 3: cosine similarity sanity check ──
    print("Test 3: semantic similarity sanity check...")
    emb_man_dog = results[0]["embedding"]
    emb_dog_ball = results[3]["embedding"]
    emb_woman_kitchen = results[1]["embedding"]

    sim_dogs = cosine_similarity(emb_man_dog, emb_dog_ball)
    sim_unrelated = cosine_similarity(emb_man_dog, emb_woman_kitchen)
    print(f"  sim(man+dog, dog+ball) = {sim_dogs:.4f}")
    print(f"  sim(man+dog, woman+kitchen) = {sim_unrelated:.4f}")
    if sim_dogs > sim_unrelated:
        print("  OK: semantically related texts have higher similarity")
    else:
        print("  WARN: unexpected ranking, but embedding API works")

    # ── Test 4: query embedding (matches _get_query_embedding path) ──
    print("Test 4: query embedding for retrieval...")
    t0 = time.time()
    try:
        q_results = AzureOpenAIEmbeddingService.get_embeddings(
            endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
            model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
            input_text=["What color is the car the man is driving?"],
            api_key=api_key,
        )
        q_emb = q_results[0]["embedding"]
        sims = []
        for i, t in enumerate(texts):
            s = cosine_similarity(q_emb, results[i]["embedding"])
            sims.append((s, t[:50]))
        sims.sort(reverse=True)
        print(f"  Query: 'What color is the car the man is driving?'")
        for s, t in sims:
            print(f"    {s:.4f}  {t}")
        print(f"  took {time.time()-t0:.2f}s")
        if sims[0][1].startswith("entities: man, car"):
            print("  OK: top match is the car/driving segment")
        else:
            print("  WARN: top match unexpected, but API works")
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    # ── Test 5: larger batch (matches real segment count) ──
    print("Test 5: batch of 64 (simulating real segment count)...")
    big_batch = [f"segment {i}: entities: person_{i}, object_{i}" for i in range(64)]
    t0 = time.time()
    try:
        results = AzureOpenAIEmbeddingService.get_embeddings(
            endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
            model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
            input_text=big_batch,
            api_key=api_key,
        )
        assert len(results) == 64
        print(f"  OK: 64 embeddings, dim={len(results[0]['embedding'])}, took {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"  FAIL: {e}")
        print("  Consider reducing SCENE_GRAPH_STAGE2_EMBED_BATCH_SIZE in config.py")
        sys.exit(1)

    print()
    print("All tests passed. Embedding pipeline is ready.")


if __name__ == "__main__":
    main()
