"""
FastAPI REST API Service for ReviewLens.
Exposes real-time NLP analysis endpoints and serves the modern frontend web UI.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Deque
from collections import defaultdict, deque
import threading
import time
import os
import io
import secrets

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Ensure package is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from reviewlens.pipeline import ReviewLensPipeline
from reviewlens.models import ReviewAnalysis, BatchAnalysisResult
from reviewlens.config import DEFAULT_ASPECT_TAXONOMY
import pandas as pd

app = FastAPI(
    title="ReviewLens API",
    description="NLP-Based Smart Product Review Analysis Microservice with Rate Limiting & Auth",
    version="1.0.0"
)

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

# Request size limiter (15 MB maximum)
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(15 * 1024 * 1024)))

@app.middleware("http")
async def request_size_limiter(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_CONTENT_LENGTH:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Payload too large. Maximum allowed size is {MAX_CONTENT_LENGTH} bytes."}
                )
        except ValueError:
            pass
    elif request.method in ("POST", "PUT", "PATCH"):
        # Handle chunked transfer-encoding or missing content-length header
        body_parts = []
        bytes_received = 0
        async for chunk in request.stream():
            bytes_received += len(chunk)
            if bytes_received > MAX_CONTENT_LENGTH:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Payload too large. Maximum allowed size is {MAX_CONTENT_LENGTH} bytes."}
                )
            body_parts.append(chunk)

        full_body = b"".join(body_parts)
        async def receive():
            return {"type": "http.request", "body": full_body, "more_body": False}
        request._receive = receive

    return await call_next(request)

# Rate limiting: 60 requests/minute per client IP
RATE_LIMIT_PER_MINUTE = int(os.getenv("REVIEWLENS_RATE_LIMIT", "60"))
_rate_limit_lock = threading.Lock()
_client_requests: Dict[str, Deque[float]] = defaultdict(deque)

def check_rate_limit(client_id: str) -> bool:
    """
    Sliding-window rate limiter keyed on client IP address.
    Prunes expired timestamps and removes inactive client entries to bound memory usage.
    """
    now = time.time()
    window = 60.0
    with _rate_limit_lock:
        # Prune inactive client keys to prevent unbounded memory growth
        stale_clients = [cid for cid, times in _client_requests.items() if not times or times[-1] < now - window]
        for cid in stale_clients:
            del _client_requests[cid]

        req_times = _client_requests[client_id]
        while req_times and req_times[0] < now - window:
            req_times.popleft()
        if len(req_times) >= RATE_LIMIT_PER_MINUTE:
            return False
        req_times.append(now)
        return True

# API Key Authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
CONFIGURED_API_KEY = os.getenv("REVIEWLENS_API_KEY", "")
REQUIRE_API_KEY = os.getenv("REVIEWLENS_REQUIRE_API_KEY", "false").lower() in ("true", "1", "yes")

async def verify_api_access(request: Request, api_key: Optional[str] = Security(API_KEY_HEADER)):
    """
    Validates API key (timing-attack resistant via secrets.compare_digest) and enforces
    sliding-window IP-based rate limiting. Rejects requests if key requirement is misconfigured.
    """
    configured_key = (os.getenv("REVIEWLENS_API_KEY", CONFIGURED_API_KEY) or "").strip()
    require_key = (os.getenv("REVIEWLENS_REQUIRE_API_KEY", "false").lower() in ("true", "1", "yes")) or REQUIRE_API_KEY

    # 1. API Key Validation
    if require_key or configured_key:
        if require_key and not configured_key:
            # If server is configured to require an API key but no key was set, reject all requests
            raise HTTPException(
                status_code=500,
                detail="Server configuration error: REVIEWLENS_REQUIRE_API_KEY is enabled, but no valid REVIEWLENS_API_KEY is configured."
            )

        auth_header = request.headers.get("Authorization", "")
        bearer_token = auth_header.replace("Bearer ", "").strip() if auth_header.startswith("Bearer ") else ""
        provided_key = (api_key or bearer_token or "").strip()

        if not provided_key or not secrets.compare_digest(provided_key, configured_key):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API Key. Provide a valid 'X-API-Key' or 'Authorization: Bearer <key>' header.",
                headers={"WWW-Authenticate": "ApiKey"}
            )

    # 2. Rate Limiting keyed on verified client IP (unverified/random keys cannot bypass)
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 60 requests per minute allowed.",
            headers={"Retry-After": "60"}
        )

    return True

pipeline = ReviewLensPipeline()


class SingleReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=15000, example="The battery life is stellar and lasts 40 hours, but the earcups feel tight.")
    rating: Optional[float] = Field(None, ge=1.0, le=5.0, example=4.0)
    title: Optional[str] = Field(None, max_length=500, example="Great battery, uncomfortable fit")
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


@app.post("/api/v1/analyze", response_model=ReviewAnalysis, dependencies=[Depends(verify_api_access)])
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


MAX_BATCH_SIZE = 100


@app.post("/api/v1/analyze-batch", response_model=BatchAnalysisResult, dependencies=[Depends(verify_api_access)])
def analyze_batch(request: BatchReviewRequest):
    """
    Analyze a batch of reviews and generate executive KPIs, SWOT, and recommendations.
    Enforces maximum batch request limit of 100 items for DoS protection.
    """
    if not request.reviews:
        raise HTTPException(status_code=400, detail="Batch reviews list cannot be empty.")
    if len(request.reviews) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds maximum limit of {MAX_BATCH_SIZE} reviews per request.")

    dicts = [r.model_dump() for r in request.reviews]
    result = pipeline.analyze_dataset(dicts)
    return result


@app.post("/api/v1/upload-csv", response_model=BatchAnalysisResult, dependencies=[Depends(verify_api_access)])
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
