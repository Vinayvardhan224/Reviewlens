# ReviewLens: Smart Shopping Platform & NLP Review Intelligence

**ReviewLens** is an independent, explainable Natural Language Processing (NLP) smart shopping platform and decision-support application. It deconstructs e-commerce consumer feedback into granular product dimensions, detects fake rating anomalies, compares electronics side-by-side with multi-dimensional radar charts, and delivers transparent, algorithmic product recommendations.

---

## 🌟 Key Capabilities

1. **Amazon-Inspired Smart Discovery & Catalogue**:
   - Modern utility header with quick department navigation (*Smartphones, Audio, Laptops, Wearables, Cameras*).
   - Instant search across product titles, brands, specifications, and descriptions.
   - Comprehensive multi-attribute filtering (Brand multiselect, INR Price slider, minimum star rating, sort order).
   - Uniform product cards featuring formatted INR prices (`₹xx,xxx`), star ratings, review counts, outbound retailer links to Amazon India, and a floating **Compare Tray**.

2. **Explainable Aspect-Based Sentiment Analysis (ABSA)**:
   - Deconstructs feedback into 8 core hardware & software dimensions:
     - 🚀 **Performance & Speed**
     - 📱 **Display & Screen**
     - 📸 **Camera & Photography**
     - 🔋 **Battery & Charging**
     - 🛡️ **Build Quality & Design**
     - 💰 **Value for Money & Price**
     - 🔊 **Audio & Sound**
     - ⚙️ **Software & Features**
   - Calculates **Net Sentiment Score (NSS = % Positive - % Negative)** per aspect from -100 to +100.
   - Extracts authentic positive and negative quote evidence snippets for full auditability.

3. **Multi-Product Side-by-Side Comparison & Radar Duel**:
   - Compares 2 or 3 products simultaneously on common hardware specifications.
   - Computes shared aspect sentiment deltas and highlights head-to-head category winners.
   - Visualizes multi-trace aspect radar polygons with zero collision.
   - Delivers transparent overall winner verdicts and rationales.

4. **Algorithmic, Defensible Recommendations**:
   - Mathematical composite scoring formula:
     $$\text{Score} = (0.30 \times \text{Star Rating \%}) + (0.45 \times \text{Normalized NSS}) + (0.25 \times \text{CSAT \%})$$
   - Category-specific winners: *Best Overall*, *Best Budget / Value Pick*, *Best Camera*, *Best Battery*, and *Best Performance*.

5. **Rating Anomaly & Authenticity Detection**:
   - Spots discrepancies between numeric stars and text sentiment (e.g. 5-star rating with severe negative complaints, or 1-star with glowing praise).

6. **Transparent Data Provenance**:
   - Prepared demo reviews and specs are explicitly labeled (`data_type='prepared_demo'`) to preserve data integrity.
   - Outbound links direct users to live Amazon India product listings for real-time market data.

---

## 🏗️ Project Architecture

```
reviewlens/
├── app.py                      # Main Streamlit discovery & analytics application
├── streamlit_app.py            # Streamlit Cloud deployment entrypoint
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation & setup instructions
├── data/
│   ├── products.csv            # Catalogue dataset (specifications, INR pricing, provenance)
│   └── reviews.csv             # Verified reviews corpus mapped to product_id
├── modules/
│   ├── data_loader.py          # Data ingestion, indexing, and INR currency formatting
│   ├── validation.py           # Dataset schema validation, ranges, and orphan review checks
│   ├── preprocessing.py        # Text cleaning, sentence splitting, and negation handling
│   ├── sentiment.py            # VADER polarity scoring, NSS formula, and anomaly detector
│   ├── aspect_analysis.py      # Rule-based aspect classification and evidence snippet extractor
│   ├── keyword_extraction.py   # TF-IDF unigrams & bigrams (positive drivers vs. pain points)
│   ├── review_summary.py       # Executive summary, buyer personas, and pros/cons synthesis
│   ├── comparison.py           # Multi-product comparison engine and delta calculator
│   └── recommendation.py       # Weighted recommendation engine and scoring formulas
├── components/
│   ├── header.py               # Amazon-style utility navigation & compare tray counter
│   ├── product_card.py         # Uniform product discovery cards
│   ├── filters.py              # Department, brand, price slider, and sorting controls
│   └── charts.py               # Plotly charts (Aspect breakdown, radar, scatter, TF-IDF bars)
└── tests/
    ├── test_data_validation.py # Schema & data integrity tests
    ├── test_sentiment.py       # Polarity, NSS, and anomaly tests
    ├── test_aspects.py         # Aspect extraction and aggregation tests
    └── test_recommendations.py # Recommendation scoring tests
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

### 2. Run the Web Application
Launch the Streamlit web dashboard locally:
```bash
streamlit run app.py
```
Or:
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run the Test Suite
Execute the comprehensive automated test suite (32 tests):
```bash
python -m unittest discover tests
```

---

## ☁️ Streamlit Cloud Deployment Guide

When deploying to [Streamlit Community Cloud](https://share.streamlit.io):
1. **Repository**: `Vinayvardhan224/reviewlens`
2. **Branch**: `main`
3. **Main file path**: Set to either `app.py` or `streamlit_app.py` (both files are present and supported).
4. Click **Deploy!**

---

## 📊 Dataset Schema

### `data/products.csv`
| Column | Type | Description |
| :--- | :--- | :--- |
| `product_id` | string | Unique product slug (e.g. `samsung-galaxy-s24-fe`) |
| `product_name` | string | Full commercial product name |
| `brand` | string | Manufacturer brand |
| `category` | string | Department (*Smartphones, Audio, Laptops, Wearables, Cameras*) |
| `price` | float | Retail price in Indian Rupees (INR) |
| `rating` | float | Customer star rating (1.0 to 5.0) |
| `review_count`| int | Total user ratings |
| `description` | string | Marketing description and key specs |
| `amazon_url` | string | Outbound link to Amazon India |
| `image_url` | string | Product image URL |
| `specifications` | JSON string | Key-value dictionary of hardware attributes |
| `data_type` | string | Provenance flag (`prepared_demo`) |

### `data/reviews.csv`
| Column | Type | Description |
| :--- | :--- | :--- |
| `review_id` | string | Unique review identifier (e.g. `REV-S24FE-0001`) |
| `product_id` | string | References `products.csv.product_id` |
| `review_text` | string | Consumer review body covering hardware dimensions |
| `review_rating`| float | Numeric rating (1.0 to 5.0) |
| `review_date` | string | Date of review publication |
| `verified_purchase` | boolean | Verified purchase flag |
| `source` | string | Origin platform tag |
| `data_type` | string | Provenance flag (`prepared_demo`) |

---

## 🛠️ Installation & NLTK Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. NLTK Lexical Resource Setup
ReviewLens requires NLTK lexicons (`vader_lexicon`, `punkt`, `punkt_tab`, `stopwords`, `wordnet`) for Aspect-Based Sentiment Analysis and tokenization. Run the dedicated setup script:
```bash
python setup_nltk.py
```

### 3. Launch the Application
```bash
streamlit run app.py
```
Both the **Smart Shopping & Product Intelligence Platform** and the **Original ReviewLens Analytics Dashboard** (with its 7 interactive tabs) are accessible through the sidebar navigation.

---

## 📜 License
MIT License. Open-source for academic and portfolio demonstration.
