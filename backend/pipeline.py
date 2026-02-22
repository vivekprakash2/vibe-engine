"""
pipeline.py
-----------
Core vibe-based song recommendation pipeline extracted from Colab notebooks.

API keys are loaded automatically from a .env file in the working directory,
or from environment variables already set in the shell/deployment platform.

Required .env variables:
    GEMINI_API_KEY
    SPOTIFY_CLIENT_ID
    SPOTIFY_CLIENT_SECRET

Usage:
    from pipeline import load_resources, recommend, recommend_from_image

    # Call once at startup (slow - loads embeddings + model)
    load_resources(
        embeddings_path="final_embeddings_by_popularity.npy",
        metadata_dir="top_500000_by_views_parquet/",
        # OR: metadata_path="final_meta.parquet",
    )

    # Text-based recommendation
    results = recommend("grindset mindset", k=10)

    # Image-based recommendation
    with open("mood.jpg", "rb") as f:
        results = recommend_from_image(f.read(), mime_type="image/jpeg", k=10)

    # Each result includes:
    #   title, artist, year, views, spotify_url,
    #   album_art, lyric_snippet, final_score,
    #   semantic_score, enriched
"""

import os
import glob
import json
import re

import numpy as np
import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types

# Load .env file automatically on module import
load_dotenv()

# ---------------------------------------------------------------------------
# Module-level state (populated by load_resources)
# ---------------------------------------------------------------------------
_embed_model: SentenceTransformer = None
_final_matrix: np.ndarray = None
_full_songs_metadata: pd.DataFrame = None
_client: genai.Client = None
_spotify: spotipy.Spotify = None


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def load_resources(
    embeddings_path: str,
    metadata_dir: str = None,
    metadata_path: str = None,
    min_lyric_length: int = 100,
    device: str = "cpu",
):
    """
    Load all heavy resources into module-level globals.
    Call this ONCE before calling recommend().

    API keys are read automatically from .env / environment variables:
        GEMINI_API_KEY
        SPOTIFY_CLIENT_ID
        SPOTIFY_CLIENT_SECRET

    Parameters
    ----------
    embeddings_path  : path to final_embeddings_by_popularity.npy
    metadata_dir     : path to folder containing .parquet chunk files
    metadata_path    : path to a single pre-built metadata .parquet file
                       (use this OR metadata_dir, not both)
    min_lyric_length : minimum character count to keep a song row (default 100)
    device           : 'cpu' or 'cuda'
    """
    global _embed_model, _final_matrix, _full_songs_metadata, _client, _spotify

    # Guard against double-loading
    if _embed_model is not None:
        print("Resources already loaded, skipping.")
        return

    # --- Gemini client ---
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. "
            "Add it to your .env file or set it as an environment variable."
        )
    _client = genai.Client()
    print("✅ Gemini client ready.")

    # --- Spotify client ---
    sp_id     = os.environ.get("SPOTIFY_CLIENT_ID", "")
    sp_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    if sp_id and sp_secret:
        _spotify = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=sp_id,
                client_secret=sp_secret,
            )
        )
        print("✅ Spotify client ready.")
    else:
        print(
            "⚠️  SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET not found in .env — "
            "album_art and spotify_url will be None."
        )

    # --- Sentence transformer ---
    print(f"Loading sentence transformer on {device.upper()}...")
    _embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print("✅ Sentence transformer loaded.")

    # --- Embeddings matrix ---
    print(f"Loading embeddings from {embeddings_path}...")
    _final_matrix = np.load(embeddings_path).astype("float32")
    print(f"✅ Embeddings loaded: {_final_matrix.shape}")

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
        f"✅ All resources loaded. "
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


def _is_close_match(a: str, b: str) -> bool:
    """Check if two strings are roughly the same after lowercasing and stripping punctuation."""
    a_clean = re.sub(r"[^\w\s]", "", a.lower()).strip()
    b_clean = re.sub(r"[^\w\s]", "", b.lower()).strip()
    return a_clean in b_clean or b_clean in a_clean


def _run_search(
    enriched: dict,
    k: int,
    semantic_weight: float,
    popularity_weight: float,
    recency_weight: float,
    max_per_artist: int,
    min_semantic: float,
    candidate_pool: int,
) -> list[dict]:
    """
    Shared search logic used by both recommend() and recommend_from_image().
    Takes an already-enriched query dict and returns ranked results with
    Spotify album art and links attached.

    Spotify lookups are batched after candidate selection to avoid making
    API calls for songs that get filtered out by diversity rules.
    """
    query_text      = enriched["rewritten_prompt"]
    extra_phrasings = enriched.get("keywords", []) + enriched.get("moods", [])

    # --- Encode query ---
    query_vec = _expand_query(query_text, extra_phrasings)

    # --- Semantic similarity ---
    semantic_scores = cosine_similarity(query_vec, _final_matrix).flatten()

    # --- Semantic floor mask ---
    floor_mask = semantic_scores >= min_semantic

    # --- Popularity score (log-normalised views) ---
    if "views" in _full_songs_metadata.columns:
        views = pd.to_numeric(_full_songs_metadata["views"], errors="coerce").fillna(0)
    else:
        views = pd.Series(np.zeros(len(_full_songs_metadata)))

    popularity = np.log10(views + 1)
    pop_min, pop_max = popularity.min(), popularity.max()
    popularity = (popularity - pop_min) / (pop_max - pop_min + 1e-9)

    # --- Recency score ---
    if "year" in _full_songs_metadata.columns:
        years = pd.to_numeric(_full_songs_metadata["year"], errors="coerce")
    else:
        years = pd.Series(np.full(len(_full_songs_metadata), np.nan))

    yr_min, yr_max = years.min(), years.max()
    recency = (years - yr_min) / (yr_max - yr_min + 1e-9)
    recency = recency.fillna(0.5)

    # --- Blended score ---
    final_scores = (
        (semantic_scores     * semantic_weight)
        + (popularity.values * popularity_weight)
        + (recency.values    * recency_weight)
    )
    final_scores[~floor_mask] = 0.0

    # --- Diversity filtering (no Spotify calls yet) ---
    candidate_indices  = final_scores.argsort()[-(k * candidate_pool):][::-1]
    results            = []
    seen_titles        = set()
    seen_artist_counts = {}

    for i in candidate_indices:
        if final_scores[i] == 0:
            continue

        row         = _full_songs_metadata.iloc[i]
        clean_title = str(row["title"]).lower().split("(")[0].strip()
        artist      = str(row.get("artist", "")).lower().strip()

        if clean_title in seen_titles:
            continue
        if seen_artist_counts.get(artist, 0) >= max_per_artist:
            continue

        seen_titles.add(clean_title)
        seen_artist_counts[artist] = seen_artist_counts.get(artist, 0) + 1

        # Append result without Spotify data yet
        results.append({
            "title":          str(row.get("title", "")),
            "artist":         str(row.get("artist", "")),
            "year":           int(years.iloc[i]) if not pd.isna(years.iloc[i]) else None,
            "views":          int(views.iloc[i]) if views.iloc[i] > 0 else None,
            "spotify_url":    None,
            "album_art":      None,
            "explanation":    None,
            "lyric_snippet":  str(row.get("lyrics_clean", ""))[:150],
            "final_score":    float(final_scores[i]),
            "semantic_score": float(semantic_scores[i]),
            "enriched":       enriched,
        })

        if len(results) >= k:
            break

    # --- Batch Spotify lookups after candidate selection ---
    # Only call Spotify for the final k results, not discarded candidates
    for r in results:
        spotify_data   = get_spotify_data(r["title"], r["artist"])
        r["spotify_url"] = spotify_data["spotify_url"]
        r["album_art"]   = spotify_data["album_art"]
        
        
    # --- Batch explanation generation (single Gemini call) ---
    explanations = _generate_explanations_batch(
        prompt=enriched["rewritten_prompt"],
        results=results,
        moods=enriched.get("moods", []),
    )
    for r, explanation in zip(results, explanations):
        r["explanation"] = explanation

    return results


# ---------------------------------------------------------------------------
# Spotify lookup
# ---------------------------------------------------------------------------

def get_spotify_data(title: str, artist: str) -> dict:
    """
    Look up a song on Spotify and return album art URL and Spotify track link.
    Validates that the returned track actually matches the search to avoid
    returning links for the wrong song.

    Returns None values gracefully if no match is found or Spotify is unavailable.

    Parameters
    ----------
    title  : song title
    artist : artist name

    Returns
    -------
    dict with keys:
        album_art    (str | None)  — URL to highest-res album cover image
        spotify_url  (str | None)  — direct link to track on Spotify
    """
    if _spotify is None:
        return {"album_art": None, "spotify_url": None}

    try:
        # Strict search first
        query   = f"track:{title} artist:{artist}"
        results = _spotify.search(q=query, type="track", limit=1)
        tracks  = results["tracks"]["items"]

        # Fallback to loose search
        if not tracks:
            query   = f"{title} {artist}"
            results = _spotify.search(q=query, type="track", limit=1)
            tracks  = results["tracks"]["items"]

        if not tracks:
            return {"album_art": None, "spotify_url": None}

        track           = tracks[0]
        returned_title  = track["name"]
        returned_artist = track["artists"][0]["name"]

        # Validate the result actually matches what we searched for
        title_match  = _is_close_match(title, returned_title)
        artist_match = _is_close_match(artist, returned_artist)

        if not title_match or not artist_match:
            print(
                f"⚠️  Spotify mismatch for '{title}' by '{artist}' — "
                f"got '{returned_title}' by '{returned_artist}'. Skipping."
            )
            return {"album_art": None, "spotify_url": None}

        images = track["album"]["images"]
        return {
            "album_art":   images[0]["url"] if images else None,
            "spotify_url": track["external_urls"]["spotify"],
        }

    except Exception as e:
        print(f"⚠️  Spotify lookup failed for '{title}' by '{artist}': {e}")
        return {"album_art": None, "spotify_url": None}


# ---------------------------------------------------------------------------
# Gemini enrichment — text
# ---------------------------------------------------------------------------

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
# Gemini enrichment — image
# ---------------------------------------------------------------------------

def enrich_prompt_from_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    model: str = "gemini-2.5-flash",
) -> dict:
    """
    Use Gemini Vision to derive a vibe query from an uploaded image,
    then return the same enrichment structure as enrich_prompt().

    Parameters
    ----------
    image_bytes : raw bytes of the uploaded image
    mime_type   : MIME type of the image (e.g. 'image/jpeg', 'image/png')
    model       : Gemini model to use

    Returns
    -------
    Same dict structure as enrich_prompt():
        rewritten_prompt, moods, themes, keywords, energy
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
        "You are a music recommendation assistant. "
        "Look at the image and describe the mood, atmosphere, and emotions it conveys. "
        "Then convert those observations into a retrieval-friendly music query. "
        "Return concise moods/themes/keywords that would help match song lyrics to this image's vibe. "
        "Keep rewritten_prompt short (<= 25 words). Avoid artist names.\n"
    )

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    response = _client.models.generate_content(
        model=model,
        contents=[image_part],
        config=types.GenerateContentConfig(
            system_instruction=instructions,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
        ),
    )
    return json.loads(response.text)


# ---------------------------------------------------------------------------
# Public recommendation functions
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
    Text prompt → ranked song recommendations with Spotify data.

    Parameters
    ----------
    prompt                : raw user vibe query (e.g. "grindset mindset")
    k                     : number of results to return (default 10)
    semantic_weight       : weight for cosine similarity score (default 0.7)
    popularity_weight     : weight for log-normalised view count (default 0.2)
    recency_weight        : weight for release year recency (default 0.1)
    max_per_artist        : max results from the same artist (default 2)
    min_semantic          : minimum cosine similarity floor (default 0.25)
    candidate_pool        : multiplier for initial candidate set size (default 5)
    use_gemini_enrichment : if False, skips Gemini and uses raw prompt (faster)

    Returns
    -------
    List of dicts, each with:
        title          (str)
        artist         (str)
        year           (int | None)
        views          (int | None)
        spotify_url    (str | None)    — direct Spotify track link
        album_art      (str | None)    — URL to album cover image
        lyric_snippet  (str)
        final_score    (float)
        semantic_score (float)
        enriched       (dict | None)   — Gemini enrichment output
    """
    assert abs(semantic_weight + popularity_weight + recency_weight - 1.0) < 1e-6, \
        "Weights must sum to 1.0"

    if use_gemini_enrichment:
        enriched = enrich_prompt(prompt)
    else:
        enriched = {
            "rewritten_prompt": prompt,
            "moods":    [],
            "themes":   [],
            "keywords": [],
            "energy":   "medium",
        }

    return _run_search(
        enriched=enriched,
        k=k,
        semantic_weight=semantic_weight,
        popularity_weight=popularity_weight,
        recency_weight=recency_weight,
        max_per_artist=max_per_artist,
        min_semantic=min_semantic,
        candidate_pool=candidate_pool,
    )


def recommend_from_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    k: int = 10,
    semantic_weight: float = 0.7,
    popularity_weight: float = 0.2,
    recency_weight: float = 0.1,
    max_per_artist: int = 2,
    min_semantic: float = 0.25,
    candidate_pool: int = 5,
) -> list[dict]:
    """
    Image upload → ranked song recommendations with Spotify data.

    Passes the image through Gemini Vision to extract vibe/mood,
    then runs the same search pipeline as recommend().

    Parameters
    ----------
    image_bytes     : raw bytes of the uploaded image
    mime_type       : MIME type (default 'image/jpeg'; also accepts 'image/png')
    k, semantic_weight, popularity_weight, recency_weight,
    max_per_artist, min_semantic, candidate_pool — same as recommend()

    Returns
    -------
    Same structure as recommend() — list of dicts with Spotify data attached.
    """
    assert abs(semantic_weight + popularity_weight + recency_weight - 1.0) < 1e-6, \
        "Weights must sum to 1.0"

    enriched = enrich_prompt_from_image(image_bytes, mime_type=mime_type)

    return _run_search(
        enriched=enriched,
        k=k,
        semantic_weight=semantic_weight,
        popularity_weight=popularity_weight,
        recency_weight=recency_weight,
        max_per_artist=max_per_artist,
        min_semantic=min_semantic,
        candidate_pool=candidate_pool,
    )
    

def _generate_explanations_batch(
    prompt: str,
    results: list[dict],
    moods: list[str],
    model: str = "gemini-2.5-flash",
) -> list[str]:
    """
    Generate one-sentence explanations for all recommended songs in a single
    Gemini call. Returns a list of explanation strings in the same order as results.
    """
    schema = {
        "type": "object",
        "properties": {
            "explanations": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["explanations"],
    }

    songs_block = "\n".join([
        f"{i+1}. \"{r['title']}\" by {r['artist']} — snippet: {r['lyric_snippet'][:100]}"
        for i, r in enumerate(results)
    ])

    content = (
        f"User vibe: \"{prompt}\"\n"
        f"Moods detected: {', '.join(moods)}\n\n"
        f"Songs recommended:\n{songs_block}\n\n"
        f"For each song, write exactly one sentence explaining why it matches "
        f"the user's vibe. Be specific, reference the mood or lyric content. "
        f"Do not start with 'This song' or 'I'. Keep each under 20 words. "
        f"Return exactly {len(results)} explanations in the same order as the songs."
    )

    try:
        response = _client.models.generate_content(
            model=model,
            contents=content,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.4,
            ),
        )
        explanations = json.loads(response.text)["explanations"]

        # Safety net — if Gemini returns wrong count, pad or trim
        if len(explanations) < len(results):
            explanations += [""] * (len(results) - len(explanations))
        return explanations[:len(results)]

    except Exception as e:
        print(f"⚠️  Batch explanation generation failed: {e}")
        return [""] * len(results)
