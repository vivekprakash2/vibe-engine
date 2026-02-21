"""
pipeline.py
-----------
Core vibe-based song recommendation pipeline extracted from Colab notebooks.

Usage:
    from pipeline import load_resources, recommend

    # Call once at startup (slow - loads embeddings + model)
    load_resources(
        embeddings_path="final_embeddings_by_popularity.npy",
        metadata_dir="top_500000_by_views_parquet/",   # folder of .parquet chunks
        # OR pass a single pre-built metadata parquet:
        # metadata_path="final_meta.parquet",
    )

    results = recommend("grindset mindset", k=10)
"""

import os
import glob
import json

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Module-level state (populated by load_resources)
# ---------------------------------------------------------------------------
_embed_model: SentenceTransformer = None
_final_matrix: np.ndarray = None
_full_songs_metadata: pd.DataFrame = None
_client: genai.Client = None


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def load_resources(
    embeddings_path: str,
    metadata_dir: str = None,
    metadata_path: str = None,
    gemini_api_key: str = None,
    min_lyric_length: int = 100,
    device: str = "cpu",
):
    """
    Load all heavy resources into module-level globals.
    Call this ONCE before calling recommend().

    Parameters
    ----------
    embeddings_path   : path to final_embeddings_by_popularity.npy
    metadata_dir      : path to folder containing .parquet chunk files
    metadata_path     : path to a single pre-built metadata .parquet file
                        (use this OR metadata_dir, not both)
    gemini_api_key    : Gemini API key. Falls back to GEMINI_API_KEY env var.
    min_lyric_length  : minimum character count to keep a song row (default 100)
    device            : 'cpu' or 'cuda'
    """
    global _embed_model, _final_matrix, _full_songs_metadata, _client

    # --- Gemini client ---
    api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    _client = genai.Client()

    # --- Sentence transformer ---
    print(f"Loading sentence transformer on {device.upper()}...")
    _embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # --- Embeddings matrix ---
    print(f"Loading embeddings from {embeddings_path}...")
    _final_matrix = np.load(embeddings_path).astype("float32")

    # --- Song metadata ---
    if metadata_path:
        print(f"Loading metadata from {metadata_path}...")
        _full_songs_metadata = pd.read_parquet(metadata_path)
    elif metadata_dir:
        parquet_files = sorted(glob.glob(f"{metadata_dir}/*.parquet"))
        print(f"Reconstructing metadata from {len(parquet_files)} parquet chunks...")
        chunks = [pd.read_parquet(f) for f in parquet_files]
        _full_songs_metadata = pd.concat(chunks, ignore_index=True)
    else:
        raise ValueError("Provide either metadata_path or metadata_dir.")

    # --- Sanity check ---
    if len(_full_songs_metadata) != _final_matrix.shape[0]:
        raise ValueError(
            f"Row mismatch: {len(_full_songs_metadata)} metadata rows "
            f"vs {_final_matrix.shape[0]} embedding rows."
        )

    # --- Filter short lyrics ---
    if "lyrics_clean" in _full_songs_metadata.columns:
        mask = _full_songs_metadata["lyrics_clean"].str.len() >= min_lyric_length
        _full_songs_metadata = _full_songs_metadata[mask].reset_index(drop=True)
        _final_matrix = _final_matrix[mask]

    # --- Clean artist names ---
    if "artist" in _full_songs_metadata.columns:
        _full_songs_metadata["artist"] = _full_songs_metadata["artist"].apply(_clean_artist)
        _full_songs_metadata["artist"] = (
            _full_songs_metadata["artist"]
            .str.split(r",|&", regex=True)
            .str[0]
            .str.strip()
        )

    print(
        f"✅ Resources loaded. "
        f"{len(_full_songs_metadata):,} songs | "
        f"{_final_matrix.shape[1]}-dim embeddings."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_artist(artist_str: str) -> str:
    """Strip the 'Lyrics - ' prefix that appears in some artist fields."""
    s = str(artist_str)
    if " Lyrics - " in s:
        return s.split(" Lyrics - ", 1)[1].strip()
    return s


def _expand_query(query: str, extra_phrasings: list = None) -> np.ndarray:
    """
    Encode query (plus any extra phrasings) and return their mean vector.
    Produces a richer, multi-angle query representation.
    """
    expansions = [query] + (extra_phrasings or [])
    vecs = _embed_model.encode(expansions, convert_to_numpy=True).astype("float32")
    return vecs.mean(axis=0, keepdims=True)


def enrich_prompt(prompt: str, model: str = "gemini-2.5-flash") -> dict:
    """
    Use Gemini to expand a raw user vibe query into structured retrieval signals.

    Returns a dict with keys:
        rewritten_prompt (str)
        moods            (list[str])
        themes           (list[str])
        keywords         (list[str])
        energy           ('low' | 'medium' | 'high')
    """
    schema = {
        "type": "object",
        "properties": {
            "rewritten_prompt": {"type": "string"},
            "moods":    {"type": "array", "items": {"type": "string"}},
            "themes":   {"type": "array", "items": {"type": "string"}},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "energy":   {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["rewritten_prompt", "moods", "themes", "keywords", "energy"],
    }

    instructions = (
        "You convert a user's song-vibe prompt into a retrieval-friendly query.\n"
        "Return concise moods/themes/keywords that would help match lyrics.\n"
        "Keep rewritten_prompt short (<= 25 words). Avoid artist names.\n"
    )

    response = _client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=instructions,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
        ),
    )
    return json.loads(response.text)


# ---------------------------------------------------------------------------
# Main recommendation function
# ---------------------------------------------------------------------------

def recommend(
    prompt: str,
    k: int = 10,
    semantic_weight: float = 0.7,
    popularity_weight: float = 0.2,
    recency_weight: float = 0.1,
    max_per_artist: int = 2,
    min_semantic: float = 0.25,
    candidate_pool: int = 5,
    use_gemini_enrichment: bool = True,
) -> list[dict]:
    """
    Full end-to-end pipeline: vibe prompt → ranked song recommendations.

    Parameters
    ----------
    prompt               : raw user vibe query (e.g. "grindset mindset")
    k                    : number of results to return (default 10)
    semantic_weight      : weight for cosine similarity score (default 0.7)
    popularity_weight    : weight for log-normalised view count (default 0.2)
    recency_weight       : weight for release year recency (default 0.1)
    max_per_artist       : max results from the same artist (default 2)
    min_semantic         : minimum cosine similarity floor (default 0.25)
    candidate_pool       : multiplier for initial candidate set size (default 5)
    use_gemini_enrichment: if False, skips Gemini and uses raw prompt (faster)

    Returns
    -------
    List of dicts, each with:
        title         (str)
        artist        (str)
        year          (int | None)
        views         (int | None)
        link          (str | None)
        lyric_snippet (str)
        final_score   (float)
        semantic_score(float)
        enriched      (dict | None)  — Gemini enrichment output
    """
    assert abs(semantic_weight + popularity_weight + recency_weight - 1.0) < 1e-6, \
        "Weights must sum to 1.0"

    # --- Step 1: Enrich prompt with Gemini ---
    enriched = None
    extra_phrasings = []

    if use_gemini_enrichment:
        enriched = enrich_prompt(prompt)
        query_text = enriched["rewritten_prompt"]
        extra_phrasings = enriched.get("keywords", []) + enriched.get("moods", [])
    else:
        query_text = prompt

    # --- Step 2: Encode query ---
    query_vec = _expand_query(query_text, extra_phrasings)

    # --- Step 3: Semantic similarity ---
    semantic_scores = cosine_similarity(query_vec, _final_matrix).flatten()

    # --- Step 4: Semantic floor mask ---
    floor_mask = semantic_scores >= min_semantic

    # --- Step 5: Popularity score (log-normalised views) ---
    views = pd.to_numeric(_full_songs_metadata.get("views", pd.Series(dtype=float)), errors="coerce").fillna(0)
    popularity = np.log10(views + 1)
    pop_min, pop_max = popularity.min(), popularity.max()
    popularity = (popularity - pop_min) / (pop_max - pop_min + 1e-9)

    # --- Step 6: Recency score ---
    years = pd.to_numeric(_full_songs_metadata.get("year", pd.Series(dtype=float)), errors="coerce")
    yr_min, yr_max = years.min(), years.max()
    recency = (years - yr_min) / (yr_max - yr_min + 1e-9)
    recency = recency.fillna(0.5)

    # --- Step 7: Blended score ---
    final_scores = (
        (semantic_scores * semantic_weight)
        + (popularity.values * popularity_weight)
        + (recency.values * recency_weight)
    )
    final_scores[~floor_mask] = 0.0

    # --- Step 8: Diversity filtering ---
    candidate_indices = final_scores.argsort()[-(k * candidate_pool):][::-1]
    results = []
    seen_titles = set()
    seen_artist_counts = {}

    for i in candidate_indices:
        if final_scores[i] == 0:
            continue

        row = _full_songs_metadata.iloc[i]
        clean_title = str(row["title"]).lower().split("(")[0].strip()
        artist = str(row.get("artist", "")).lower().strip()

        if clean_title in seen_titles:
            continue
        if seen_artist_counts.get(artist, 0) >= max_per_artist:
            continue

        seen_titles.add(clean_title)
        seen_artist_counts[artist] = seen_artist_counts.get(artist, 0) + 1

        results.append({
            "title":          str(row.get("title", "")),
            "artist":         str(row.get("artist", "")),
            "year":           int(years.iloc[i]) if not pd.isna(years.iloc[i]) else None,
            "views":          int(views.iloc[i]) if views.iloc[i] > 0 else None,
            "link":           row.get("link", None),
            "lyric_snippet":  str(row.get("lyrics_clean", ""))[:150],
            "final_score":    float(final_scores[i]),
            "semantic_score": float(semantic_scores[i]),
            "enriched":       enriched,
        })

        if len(results) >= k:
            break

    return results
