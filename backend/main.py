"""
main.py
-------
FastAPI backend for the vibe-based song recommendation engine.

Endpoints:
    GET  /health                — health check
    POST /recommend             — text prompt → song recommendations
    POST /recommend-from-image  — image upload → song recommendations

Run locally:
    uvicorn main:app --reload

API docs (auto-generated):
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Optional
import os

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import load_resources, recommend, recommend_from_image


# ---------------------------------------------------------------------------
# Lifespan — load all heavy resources once when the server starts
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs load_resources() on server startup.
    All API keys and file paths are read from environment variables / .env.
    The pipeline stays in memory for the lifetime of the server process.
    """
    embeddings_path = os.environ.get("EMBEDDINGS_PATH")
    metadata_path   = os.environ.get("METADATA_PATH")
    metadata_dir    = os.environ.get("METADATA_DIR")

    if not embeddings_path:
        raise EnvironmentError(
            "EMBEDDINGS_PATH not set. Add it to your .env file."
        )
    if not metadata_path and not metadata_dir:
        raise EnvironmentError(
            "Either METADATA_PATH or METADATA_DIR must be set in your .env file."
        )

    load_resources(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        metadata_dir=metadata_dir,
    )
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Vibe Engine API",
    description="Song recommendations based on vibes, moods, and images.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS — allows the React frontend to call this API
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # fallback
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TextRequest(BaseModel):
    prompt: str
    k: int = 10

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"prompt": "grindset mindset", "k": 10}
            ]
        }
    }


class EnrichedQuery(BaseModel):
    rewritten_prompt: str
    moods:            list[str]
    themes:           list[str]
    keywords:         list[str]
    energy:           str


class SongResult(BaseModel):
    title:          str
    artist:         str
    year:           Optional[int]
    views:          Optional[int]
    spotify_url:    Optional[str]
    album_art:      Optional[str]
    lyric_snippet:  str
    final_score:    float
    semantic_score: float
    enriched:       Optional[EnrichedQuery]
    explanation:    Optional[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
def health():
    """Check that the server is running and resources are loaded."""
    return {"status": "ok"}


@app.post(
    "/recommend",
    response_model=list[SongResult],
    tags=["Recommendations"],
    summary="Text prompt → song recommendations",
)
def recommend_from_text(body: TextRequest):
    """
    Accept a vibe prompt and return ranked song recommendations.

    - **prompt**: freeform vibe description, e.g. `"late night drive, feeling nostalgic"`
    - **k**: number of results to return (default 10)
    """
    if not body.prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt cannot be empty.")

    if body.k < 1 or body.k > 50:
        raise HTTPException(status_code=422, detail="k must be between 1 and 50.")

    try:
        results = recommend(body.prompt, k=body.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No recommendations found for this prompt. Try a different vibe."
        )

    return results


@app.post(
    "/recommend-from-image",
    response_model=list[SongResult],
    tags=["Recommendations"],
    summary="Image upload → song recommendations",
)
async def recommend_from_image_endpoint(
    file: UploadFile = File(..., description="JPEG or PNG image"),
    k: int = Query(default=10, ge=1, le=50, description="Number of results to return"),
):
    """
    Accept an image upload and return ranked song recommendations
    based on the mood and vibe Gemini Vision extracts from the image.

    - **file**: JPEG or PNG image (album art, mood board, photo, etc.)
    - **k**: number of results to return (default 10)
    """
    allowed_types = {"image/jpeg", "image/jpg", "image/png"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{file.content_type}'. Please upload a JPEG or PNG."
        )

    try:
        img_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if len(img_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        results = recommend_from_image(
            img_bytes,
            mime_type=file.content_type or "image/jpeg",
            k=k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No recommendations found for this image. Try a different one."
        )

    return results