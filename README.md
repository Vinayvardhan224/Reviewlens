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
   - Hardened credential authentication with PBKDF2-HMAC-SHA256 (100,000 iterations).
   - Enforced 8-character password policy, account lockout protection after consecutive failed attempts, and secure OTP generation.
   - Transparent sandbox demo disclosures for Google OAuth and SMS simulation.

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
│   ├── evaluation.py           # Stratified 5-fold CV baseline benchmarking
│   └── live_analyzer.py        # Real-time interactive diagnostic lab
│
├── components/                 # Reusable Streamlit UI Components
│   ├── header.py               # Amazon-style utility navigation & comparison tray
│   ├── product_card.py         # Uniform product cards with INR pricing
│   ├── filters.py              # Department, brand, price slider, and sort controls
│   └── charts.py               # Plotly aspect charts, radars, and TF-IDF visualizations
│
├── data/                       # Verified Corpora & Data Storage
│   ├── products.csv            # Catalogue dataset (12 electronics products)
│   ├── reviews.csv             # 101 verified reviews balanced across 1 to 5 stars
│   └── sample_echo_reviews.csv # Curated Amazon Echo corpus (30 reviews)
│
└── tests/                      # Automated Test Suite (69 unit tests)
    ├── test_cleaner.py         # Text preprocessing & contraction expansion tests
    ├── test_vader_analyzer.py  # VADER scoring, domain lexicon & rating mapping tests
    ├── test_spam_detector.py   # Spam heuristics & rating mismatch tests
    ├── test_aspect_extractor.py# Aspect taxonomy & classification tests
    ├── test_pipeline.py        # End-to-end pipeline & batch safety tests
    ├── test_data_validation.py # Dataset schema & range validation tests
    ├── test_sentiment.py       # Polarity, NSS, and contradiction tests
    ├── test_aspects.py         # Aspect extraction & sentiment aggregation tests
    ├── test_recommendations.py # Recommendation algorithm tests
    └── test_auth.py            # Password hashing, PBKDF2 migration & lockout tests
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
Open your browser at `http://localhost:8501`. Both the **Smart Shopping & Product Intelligence Platform** and the **Original Analytics Dashboard** are fully accessible.

#### FastAPI REST Microservice
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Modern Frontend UI: `http://localhost:8000/`
- Health check: `http://localhost:8000/api/v1/health`

#### Terminal CLI Utility
```bash
# Analyze a single review
python cli.py analyze --text "The battery life is phenomenal, easily lasting 2 days." --rating 5.0

# Batch process a dataset
python cli.py batch --input data/reviews.csv --output results.json
```

---

## 🧪 Automated Testing

ReviewLens includes a comprehensive test suite of **69 automated unit tests** covering data validation, preprocessing, sentiment analysis, aspect extraction, spam detection, recommendation math, and authentication security.

Run all tests:
```bash
python -m unittest discover tests
```

Expected output:
```
Ran 69 tests in 2.2s

OK
```

---

## 🐳 Docker Deployment

The application includes production-ready Docker and Docker Compose definitions configured to run as a non-privileged `appuser` (UID 1000).

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

In accordance with transparent research practices, ReviewLens evaluates its rule-based VADER + domain lexicon engine against supervised baseline models (`modules/evaluation.py`):

1. **Supervised Baseline**: TF-IDF (1-2 ngrams, 5,000 features) + Logistic Regression evaluated via **5-Fold Stratified Cross-Validation**.
2. **Lexicon Grounding**: VADER compound thresholds ($> +0.05$ positive, $< -0.05$ negative) supplemented by an explicit domain-specific lexicon covering hardware, battery, audio, and display terminology.
3. **Data Provenance**: Product catalogue specifications and reviews are curated demonstration datasets (`data_type='prepared_demo'`) specifically balanced across 1-star to 5-star ratings (totaling 101 reviews across 12 products) to ensure realistic negative sentiment extraction, aspect dispute identification, and unbiased evaluation.

---

## 📜 License & Acknowledgments

- **License**: MIT License. Open-source for academic research, education, and portfolio demonstration.
- **Author**: Vinay Vardhan / ReviewLens Team.
- **Repository**: [https://github.com/Vinayvardhan224/reviewlens](https://github.com/Vinayvardhan224/reviewlens)
