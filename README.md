# ReviewLens: An NLP-Based Smart Product Review Analysis System

[![ReviewLens Test Suite](https://github.com/Vinayvardhan224/reviewlens/actions/workflows/python-tests.yml/badge.svg)](https://github.com/Vinayvardhan224/reviewlens/actions)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

**ReviewLens** is an explainable Natural Language Processing (NLP) intelligence platform and decision-support system designed to analyze consumer product reviews. It deconstructs raw, unstructured feedback into granular aspect dimensions, identifies rating-sentiment anomalies and suspicious review patterns, compares products side-by-side with multi-dimensional radar visualizations, and provides transparent, mathematically auditable buying recommendations.

Developed by **Vinay Vardhan / ReviewLens Team**.

---

## 🌟 Key Capabilities

1. **Amazon-Inspired Smart Discovery & Catalogue**:
   - Modern utility header with department navigation (*Smartphones, Audio, Laptops, Wearables, Cameras*).
   - Fast full-text search across titles, brands, specifications, and descriptions.
   - Multi-attribute filtering (Brand, INR Price slider, minimum star rating, sort criteria).
   - Uniform product cards with formatted INR prices (`₹xx,xxx`), star ratings, review counts, outbound retailer links to Amazon India, and a live **Compare Tray**.

2. **Explainable Aspect-Based Sentiment Analysis (ABSA)**:
   - Deconstructs reviews into 8 core hardware and software dimensions:
     - 🚀 **Performance & Speed**
     - 📱 **Display & Screen**
     - 📸 **Camera & Photography**
     - 🔋 **Battery & Charging**
     - 🛡️ **Build Quality & Design**
     - 💰 **Value for Money & Price**
     - 🔊 **Audio & Sound**
     - ⚙️ **Software & Features**
   - Calculates **Net Sentiment Score (NSS = % Positive - % Negative)** per aspect spanning from -100 to +100.
   - Extracts authentic positive and negative quote evidence snippets for full auditability.

3. **Multi-Product Side-by-Side Comparison & Radar Duel**:
   - Compare up to 3 products simultaneously on common hardware specifications.
   - Computes shared aspect sentiment deltas and highlights head-to-head category winners.
   - Visualizes multi-trace aspect radar polygons with zero visual collision.
   - Generates transparent overall winner verdicts and rationales.

4. **Algorithmic, Defensible Recommendations**:
   - Mathematical composite scoring formula:
     $$\text{Score} = (0.30 \times \text{Star Rating \%}) + (0.45 \times \text{Normalized NSS}) + (0.25 \times \text{CSAT \%})$$
   - Category-specific winners: *Best Overall*, *Best Budget / Value Pick*, *Best Camera*, *Best Battery*, and *Best Performance*.

5. **Spam & Rating Contradiction Detection**:
   - Multi-signal heuristic spam detector (extreme brevity, repetitive text patterns, suspicious URL detection).
   - Rating anomaly detection that flags numeric-sentiment contradictions (e.g. 5-star rating with severe negative complaints, or 1-star with glowing praise) without blindly discarding legitimate reviews.

6. **Original Analytics Dashboard & Deep NLP Lab**:
   - Interactive 7-tab deep analytics dashboard for exploratory data analysis.
   - Latent Dirichlet Allocation (LDA) topic modeling with pyLDAvis visualization.
   - Emotion classification (Joy, Sadness, Anger, Fear, Surprise, Trust).
   - AI-generated review response drafts tailored to sentiment and review tone.
   - Strategic SWOT matrix synthesis (Strengths, Weaknesses, Opportunities, Threats).

7. **Production Security & Authentication**:
   - Hardened credential authentication with PBKDF2-HMAC-SHA256 (600,000 iterations meeting OWASP standards).
   - Automatic on-the-fly hash migration from legacy algorithms on authenticated login.
   - Real Google OAuth PKCE authorization integration (requires `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`; cleanly disabled with setup guidance when unconfigured; no fake or simulated accounts).
   - Genuine SMS OTP verification (requires Twilio configuration; zero plaintext on-screen OTP exposure).
   - No pre-seeded default demo accounts or hardcoded credentials in production storage.
   - Enforced password complexity (min 8 chars, uppercase, lowercase, number, special character) and progressive lockout protection.

8. **API Protection & Rate Limiting**:
   - Sliding-window rate limiter protecting API endpoints against DoS (60 req/min per client).
   - Enforced request payload size protection (15 MB maximum payload).
   - Batch request size cap (maximum 100 reviews per batch).
   - API Key authentication via `X-API-Key` or `Authorization: Bearer <key>` headers.

---

## 🏗️ Dual Architecture & Project Structure

ReviewLens is architected with a decoupled design: a core reusable NLP engine (`reviewlens`), a Streamlit interactive web application (`app.py`), a production-ready FastAPI REST service (`api.py`), and a command-line interface (`cli.py`).

```
reviewlens/
├── app.py                      # Streamlit interactive smart shopping & analytics platform
├── streamlit_app.py            # Streamlit Cloud deployment entrypoint
├── api.py                      # FastAPI REST microservice (Swagger at /docs)
├── cli.py                      # Terminal CLI tool for batch review processing
├── index.html                  # Standalone modern web UI served by FastAPI
├── setup_nltk.py               # Lexical corpora setup utility (VADER, WordNet, punkt)
├── build_datasets.py           # Dataset generator ensuring balanced 1-5 star distributions
├── setup.py                    # Package build & distribution setup
├── requirements.txt            # Python dependencies
├── requirements-dev.txt        # Development dependencies (pytest, httpx, flake8, coverage)
├── requirements-lock.txt       # Pinned production lockfile for reproducible Docker/CI builds
├── Dockerfile                  # Production container definition (non-root appuser)
├── docker-compose.yml          # Compose orchestration file
│
├── reviewlens/                 # Core NLP Engine
│   ├── pipeline.py             # ReviewLensPipeline orchestrating all NLP stages
│   ├── models.py               # Pydantic data models & typing definitions
│   ├── config.py               # Aspect taxonomy & sentiment thresholds
│   ├── preprocessing/          # Text cleaner, contraction expansion, emoji normalizer
│   ├── sentiment/              # VADER analyzer with domain lexicon & phrase matching
│   ├── aspect_extractor/       # Rule-based aspect classification engine
│   ├── spam_detector/          # Multi-signal spam & anomaly detector
│   ├── topic_modeler/          # Gensim/sklearn LDA topic modeling
│   ├── emotion_analyzer/       # Plutchik/Ekman rule-based emotion classification
│   ├── response_generator/     # Contextual AI review reply generator
│   ├── swot_generator/         # Automated SWOT matrix generator
│   ├── comparison.py           # Product comparison & delta calculator
│   └── dashboard.py            # Original analytics dashboard interface
│
├── modules/                    # Application Integration Modules
│   ├── auth.py                 # Hardened PBKDF2 authentication & demo accounts
│   ├── data_loader.py          # Data ingestion, indexing, and INR currency formatting
│   ├── validation.py           # Dataset schema validation and orphan review checks
│   ├── sentiment.py            # VADER polarity scoring & NSS calculations
│   ├── aspect_analysis.py      # Category-specific aspect analysis
│   ├── comparison.py           # Multi-product comparison engine
│   ├── recommendation.py       # Weighted recommendation scoring
│   ├── evaluation.py           # Stratified 5-fold CV baseline & SST-2 benchmarking
│   └── live_analyzer.py        # Real-time interactive diagnostic lab
│
├── components/                 # Reusable Streamlit UI Components
│   ├── header.py               # Amazon-style utility navigation & comparison tray
│   ├── product_card.py         # Uniform product cards with INR pricing
│   ├── filters.py              # Department, brand, price slider, and sort controls
│   └── charts.py               # Plotly aspect charts, radars, and TF-IDF visualizations
│
├── data/                       # Curated Datasets & Storage
│   ├── products.csv            # Catalogue dataset (12 electronics products, data_type: prepared_demo)
│   ├── reviews.csv             # 101 curated demo reviews across 12 products (data_type: prepared_demo)
│   ├── sample_echo_reviews.csv # Curated Amazon Echo corpus (12 reviews, data_type: prepared_demo)
│   ├── benchmark_sentiment.csv # 50 hand-labelled sanity check reviews (data_type: labelled_benchmark)
│   ├── public_sst2_benchmark.csv # 872 genuine public benchmark sentences from Stanford SST-2 / GLUE (data_type: public_labelled_benchmark)
│   └── users.json              # Local credential store (empty on fresh clone, no seeded demo accounts)
│
└── tests/                      # Automated Regression Test Suite (86 unit tests)
    ├── test_absa.py            # Aspect taxonomy & categorization tests
    ├── test_api.py             # FastAPI auth, rate limiting, and payload size tests
    ├── test_aspects.py         # Aspect extraction & sentiment aggregation tests
    ├── test_audit_fixes.py     # End-to-end audit regression checks
    ├── test_auth.py            # PBKDF2 600k, hash migration, lockout & OAuth/SMS tests
    ├── test_cleaner.py         # Text preprocessing & contraction expansion tests
    ├── test_comparison.py      # Product comparison & delta calculator tests
    ├── test_cross_interface_consistency.py # Pipeline vs Live Analyzer consistency tests
    ├── test_data_validation.py # Strict CSV schema, non-numeric price/rating & injection tests
    ├── test_evaluation_and_topics.py # Benchmark evaluation & topic modeling tests
    ├── test_navigation_flow.py # Application navigation state machine tests
    ├── test_new_features.py    # Live diagnostic, preference matching & timeline drift tests
    ├── test_pipeline.py        # End-to-end ReviewLensPipeline integration tests
    ├── test_recommendations.py # Mathematical recommendation scoring tests
    ├── test_scraper.py         # Data scraping fallback & sanitation tests
    ├── test_sentiment.py       # VADER polarity & Net Sentiment Score tests
    └── test_spam_detector.py   # Multi-signal spam heuristics & thread safety tests
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/Vinayvardhan224/reviewlens.git
cd reviewlens
pip install -r requirements.txt
```

### 2. Download NLTK Corpora
Initialize required NLTK lexical resources (`vader_lexicon`, `punkt`, `punkt_tab`, `stopwords`, `wordnet`):
```bash
python setup_nltk.py
```

### 3. Launch the Applications

#### Streamlit Web Platform
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. ReviewLens defaults directly to **Review Intelligence & Analysis Hub** for a primary NLP research experience, with the Storefront Catalogue accessible in secondary navigation.

#### FastAPI REST Microservice
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Modern Frontend UI: `http://localhost:8000/`
- Health check: `http://localhost:8000/api/v1/health`
- Supported Aspects: `http://localhost:8000/api/v1/aspects`
- Analyze Review (protected): `POST /api/v1/analyze` (Rate limited: 60 req/min)

#### Terminal CLI Utility
```bash
# Analyze a single review
python cli.py analyze --text "The battery life is phenomenal, easily lasting 2 days." --rating 5.0

# Batch process a dataset
python cli.py batch --input data/reviews.csv --output results.json
```

---

## 🧪 Automated Testing

ReviewLens includes a comprehensive test suite of **86 automated unit tests** across 17 test modules covering data validation, preprocessing, sentiment analysis, aspect extraction, spam detection, recommendation math, API security, PBKDF2 authentication, and empirical benchmark evaluation.

Run all tests:
```bash
python -m unittest discover tests
```

Expected output:
```
Ran 86 tests in ~17.0s

OK
```

---

## 🐳 Docker Deployment

The application includes production-ready Docker and Docker Compose definitions configured to run as a non-privileged `appuser` (UID 1000). Reproducible builds are guaranteed via `requirements-lock.txt`.

```bash
# Build and run both Streamlit and FastAPI services
docker compose up --build

# Run in detached background mode
docker compose up -d
```

- **Streamlit Platform**: `http://localhost:8501`
- **FastAPI Microservice**: `http://localhost:8000`

---

## 📊 Evaluation & Benchmarking Methodology

In accordance with transparent research practices, ReviewLens employs a **3-tier evaluation framework** that strictly distinguishes demonstration datasets from ground-truth evaluation corpora (`modules/evaluation.py`):

1. **Curated Demonstration Corpus** (`data/reviews.csv`, exactly 101 reviews across 12 products: 24 negative [1-2 stars], 8 neutral [3 stars], and 69 positive [4-5 stars], tagged `data_type='prepared_demo'`):
   - Measures agreement between user star ratings and VADER polarity predictions (**76.2% agreement**).
   - Demonstrates multi-aspect trade-offs and edge cases across consumer electronics.
2. **Small Hand-Labelled Sanity Check Benchmark** (`data/benchmark_sentiment.csv`, 50 hand-labelled review sentences: 23 positive, 17 negative, 10 neutral, with 7 spam heuristic markers, tagged `data_type='labelled_benchmark'`):
   - Fast, local regression test suite ensuring code modifications do not introduce silent model regressions.
   - Evaluates VADER rule-based classification against a 5-fold cross-validated TF-IDF + Logistic Regression baseline on identical sentences.
3. **Official Public Benchmark: Stanford Sentiment Treebank (SST-2 / GLUE Dev Set)** (`data/public_sst2_benchmark.csv`, 872 independent evaluation sentences: 444 positive, 428 negative, tagged `data_type='public_labelled_benchmark'`):
   - Ground truth evaluation from standard peer-reviewed academic NLP literature (Socher et al., Stanford NLP Group / GLUE benchmark).
   - **VADER Directional Accuracy**: **67.5%** on polarized sentences (55.6% exact 3-class accuracy including neutral predictions).
   - **Supervised Baseline (TF-IDF + Logistic Regression, 5-Fold Stratified CV)**: **65.5% accuracy**, **0.653 macro F1**.
   - Proves empirical generalization against genuine public evaluation data without author bias.
4. **Dynamic Evaluation**: All evaluation metrics (Accuracy, Precision, Recall, Macro F1, and Confusion Matrices) are computed dynamically via scikit-learn without hardcoded or fabricated numbers.
5. **High-Performance Rate Limiter**: The FastAPI REST microservice includes an amortized $O(1)$ sliding-window rate limiter with throttled periodic cleanup, preventing memory unboundedness and timing attacks.

---

## 📜 License & Acknowledgments

- **License**: MIT License. Open-source for academic research, education, and portfolio demonstration.
- **Author**: Vinay Vardhan / ReviewLens Team.
- **Repository**: [https://github.com/Vinayvardhan224/reviewlens](https://github.com/Vinayvardhan224/reviewlens)
