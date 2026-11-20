# Production Dockerfile for ReviewLens
FROM python:3.12-slim

# Set working directory and environment variables
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8501

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements from pinned lockfile for reproducible production builds
COPY requirements-lock.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-lock.txt && \
    python -m spacy download en_core_web_sm

# Pre-download required NLTK corpora during build for instant container startup
RUN python -c "import nltk; \
    nltk.download('vader_lexicon', quiet=True); \
    nltk.download('punkt', quiet=True); \
    nltk.download('punkt_tab', quiet=True); \
    nltk.download('stopwords', quiet=True); \
    nltk.download('wordnet', quiet=True)"

# Copy project files
COPY . .

# Create non-root system user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose ports for Streamlit (8501) and FastAPI (8000)
EXPOSE 8501
EXPOSE 8000

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command launches the Streamlit Interactive Dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
