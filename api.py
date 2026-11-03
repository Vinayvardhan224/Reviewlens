"""
FastAPI REST API Service for ReviewLens.
Exposes real-time NLP analysis endpoints and serves the modern frontend web UI.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

# Ensure package is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from reviewlens.pipeline import ReviewLensPipeline
from reviewlens.models import ReviewAnalysis, BatchAnalysisResult
from reviewlens.config import DEFAULT_ASPECT_TAXONOMY
import pandas as pd
import io

app = FastAPI(
    title="ReviewLens API",
    description="NLP-Based Smart Product Review Analysis Microservice",
    version="1.0.0"
)

import os

# Configurable origins for production security
ALLOWED_ORIGINS = os.getenv(
    "REVIEWLENS_ALLOWED_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

pipeline = ReviewLensPipeline()


class SingleReviewRequest(BaseModel):
    text: str = Field(..., example="The battery life is stellar and lasts 40 hours, but the earcups feel tight.")
    rating: Optional[float] = Field(None, ge=1.0, le=5.0, example=4.0)
    title: Optional[str] = Field(None, example="Great battery, uncomfortable fit")
    date: Optional[str] = Field(None, example="2026-03-15")
    verified_purchase: Optional[bool] = Field(None, example=True)


class BatchReviewRequest(BaseModel):
    reviews: List[SingleReviewRequest]


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the modern frontend UI page directly at the root URL."""
    html_file = BASE_DIR / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    return {
        "service": "ReviewLens NLP Intelligence Service",
        "status": "operational",
        "version": "1.0.0",
        "documentation": "/docs"
    }


@app.get("/api/v1/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ReviewLens"}


@app.get("/api/v1/aspects")
def get_supported_aspects() -> Dict[str, List[str]]:
    """Retrieve all supported aspect categories and seed keywords."""
    return DEFAULT_ASPECT_TAXONOMY


@app.post("/api/v1/analyze", response_model=ReviewAnalysis)
def analyze_review(request: SingleReviewRequest):
    """
    Analyze a single review in real-time.
    Computes sentiment, spam score, dominant emotion, and aspect-based sentiments.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")

    result = pipeline.analyze_single(
        text=request.text,
        rating=request.rating,
        title=request.title,
        date=request.date,
        verified_purchase=request.verified_purchase
    )
    return result


@app.post("/api/v1/analyze-batch", response_model=BatchAnalysisResult)
def analyze_batch(request: BatchReviewRequest):
    """
    Analyze a batch of reviews and generate executive KPIs, SWOT, and recommendations.
    """
    if not request.reviews:
        raise HTTPException(status_code=400, detail="Batch reviews list cannot be empty.")

    dicts = [r.model_dump() for r in request.reviews]
    result = pipeline.analyze_dataset(dicts)
    return result


@app.post("/api/v1/upload-csv", response_model=BatchAnalysisResult)
async def upload_csv_and_analyze(
    file: UploadFile = File(...),
    text_column: str = "text",
    rating_column: Optional[str] = "rating"
):
    """
    Upload a CSV review dataset and receive complete business intelligence summary.
    Includes strict validation for file size, row count, schema, and column data types.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are supported via this endpoint.")

    contents = await file.read()

    # 1. Size validation (max 15 MB)
    MAX_SIZE = 15 * 1024 * 1024
    if len(contents) > MAX_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum allowed size is 15 MB (uploaded {len(contents) / (1024*1024):.1f} MB).")

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is completely empty.")

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV format: {str(e)}")

    # 2. Row count validation
    if df.empty or len(df) == 0:
        raise HTTPException(status_code=400, detail="The uploaded CSV contains 0 rows of data.")
    if len(df) > 50000:
        raise HTTPException(status_code=400, detail=f"The uploaded CSV exceeds maximum limit of 50,000 rows (found {len(df)} rows).")

    # 3. Schema & column existence
    if text_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Required text column '{text_column}' not found. Available columns: {list(df.columns)}"
        )

    # 4. Check that text column is not all null
    valid_text_count = df[text_column].dropna().astype(str).str.strip().ne("").sum()
    if valid_text_count == 0:
        raise HTTPException(status_code=400, detail=f"Column '{text_column}' contains no readable review text.")

    # 5. Rating column validation
    if rating_column and rating_column in df.columns:
        num_ratings = pd.to_numeric(df[rating_column], errors="coerce")
        out_of_bounds = ((num_ratings < 1.0) | (num_ratings > 5.0)).sum()
        if out_of_bounds > 0:
            raise HTTPException(status_code=400, detail=f"Rating column '{rating_column}' contains {out_of_bounds} values outside the valid 1.0-5.0 range.")

    result = pipeline.analyze_dataset(df, text_column=text_column, rating_column=rating_column)
    return result
