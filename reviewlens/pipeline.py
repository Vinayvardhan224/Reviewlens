"""
Master Orchestration Pipeline for ReviewLens.
Coordinates text normalization, spam filtering, sentiment analysis, emotion detection,
aspect-based sentiment analysis (ABSA), TF-IDF keyword extraction, LDA topic modeling,
and executive insight synthesis.
"""

from typing import List, Dict, Any, Union, Optional
import pandas as pd
from reviewlens.preprocessing.cleaner import clean_text
from reviewlens.spam_detector.detector import SpamDetector
from reviewlens.sentiment.vader_analyzer import SentimentAnalyzer
from reviewlens.sentiment.emotion_analyzer import EmotionAnalyzer
from reviewlens.absa.aspect_extractor import AspectExtractor
from reviewlens.absa.aspect_scorer import AspectScorer
from reviewlens.topics.tfidf_keywords import KeywordExtractor
from reviewlens.topics.topic_modeler import TopicModeler
from reviewlens.insights.executive_summary import ExecutiveInsightsEngine
from reviewlens.models import (
    ReviewInput,
    ReviewAnalysis,
    BatchAnalysisResult,
    AspectMention,
    ExecutiveSummary,
)


class ReviewLensPipeline:
    """Unified NLP pipeline for end-to-end product review intelligence."""

    def __init__(
        self,
        custom_aspect_taxonomy: Optional[Dict[str, List[str]]] = None,
        custom_sentiment_lexicon: Optional[Dict[str, float]] = None,
        n_topics: int = 4
    ):
        self.sentiment_analyzer = SentimentAnalyzer(custom_lexicon=custom_sentiment_lexicon)
        self.emotion_analyzer = EmotionAnalyzer()
        self.spam_detector = SpamDetector()
        self.aspect_extractor = AspectExtractor(sentiment_analyzer=self.sentiment_analyzer)
        self.aspect_scorer = AspectScorer()
        self.keyword_extractor = KeywordExtractor()
        self.topic_modeler = TopicModeler(n_topics=n_topics)
        self.insights_engine = ExecutiveInsightsEngine()

    def analyze_single(
        self,
        text: str,
        rating: Optional[float] = None,
        title: Optional[str] = None,
        date: Optional[str] = None,
        verified_purchase: Optional[bool] = None,
        review_id: Optional[str] = None,
        check_duplicate: bool = False
    ) -> ReviewAnalysis:
        """
        Analyze a single product review end-to-end.
        
        Args:
            text: Raw review text.
            rating: Optional user star rating (1-5).
            title: Optional review headline.
            date: Optional review timestamp or date string.
            verified_purchase: Optional verification flag.
            review_id: Optional unique identifier.
            check_duplicate: Whether to compare against previously seen reviews.
            
        Returns:
            ReviewAnalysis object.
        """
        full_raw_text = f"{title}. {text}" if title else text
        cleaned = clean_text(full_raw_text)

        # 1. Sentiment Polarity
        sentiment = self.sentiment_analyzer.analyze(cleaned)

        # 2. Emotion Detection
        dominant_emotion, emotion_scores = self.emotion_analyzer.analyze(cleaned, sentiment.compound)
        sentiment.primary_emotion = dominant_emotion
        sentiment.emotion_scores = emotion_scores

        # 3. Spam & Fake Review Detection
        spam = self.spam_detector.analyze(
            text=full_raw_text,
            rating=rating,
            sentiment_compound=sentiment.compound,
            check_duplicate=check_duplicate
        )

        # 4. Aspect-Based Sentiment Analysis
        aspects = self.aspect_extractor.extract_from_review(full_raw_text)
        detected_categories = list(dict.fromkeys([a.category for a in aspects]))

        return ReviewAnalysis(
            id=review_id,
            original_text=full_raw_text,
            cleaned_text=cleaned,
            rating=rating,
            predicted_rating=sentiment.predicted_rating,
            sentiment=sentiment,
            spam=spam,
            aspects=aspects,
            detected_aspects=detected_categories,
            date=date,
            verified_purchase=verified_purchase
        )

    def analyze_dataset(
        self,
        data: Union[List[Dict[str, Any]], pd.DataFrame, List[str]],
        text_column: str = "text",
        rating_column: Optional[str] = "rating",
        date_column: Optional[str] = "date",
        verified_column: Optional[str] = "verified_purchase",
        id_column: Optional[str] = "id",
        title_column: Optional[str] = "title"
    ) -> BatchAnalysisResult:
        """
        Run complete NLP analysis on a collection of reviews.
        
        Args:
            data: List of review dicts, pandas DataFrame, or list of review strings.
            text_column: Name of text column if DataFrame or dict.
            rating_column: Name of star rating column if available.
            date_column: Name of date column if available.
            verified_column: Name of verified purchase flag column.
            id_column: Name of ID column.
            title_column: Name of title/headline column.
            
        Returns:
            BatchAnalysisResult with comprehensive metrics, topics, keywords, and SWOT.
        """
        self.spam_detector.reset_session()

        # Normalize input to list of review inputs
        review_inputs: List[ReviewInput] = []
        if isinstance(data, pd.DataFrame):
            for idx, row in data.iterrows():
                r_text = str(row[text_column]) if text_column in row and pd.notna(row[text_column]) else ""
                r_rating = float(row[rating_column]) if rating_column and rating_column in row and pd.notna(row[rating_column]) else None
                r_date = str(row[date_column]) if date_column and date_column in row and pd.notna(row[date_column]) else None
                r_verified = bool(row[verified_column]) if verified_column and verified_column in row and pd.notna(row[verified_column]) else None
                r_id = str(row[id_column]) if id_column and id_column in row and pd.notna(row[id_column]) else str(idx)
                r_title = str(row[title_column]) if title_column and title_column in row and pd.notna(row[title_column]) else None
                review_inputs.append(
                    ReviewInput(
                        id=r_id, text=r_text, title=r_title, rating=r_rating,
                        date=r_date, verified_purchase=r_verified
                    )
                )
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, str):
                    review_inputs.append(ReviewInput(id=str(idx), text=item))
                elif isinstance(item, dict):
                    review_inputs.append(
                        ReviewInput(
                            id=str(item.get(id_column or "id", idx)),
                            text=str(item.get(text_column, "")),
                            title=item.get(title_column or "title"),
                            rating=float(item[rating_column]) if rating_column and item.get(rating_column) is not None else None,
                            date=str(item.get(date_column or "date")) if date_column and item.get(date_column) else None,
                            verified_purchase=item.get(verified_column or "verified_purchase")
                        )
                    )

        # 1. Process each individual review
        analyzed_reviews: List[ReviewAnalysis] = []
        all_aspect_mentions: List[AspectMention] = []

        for item in review_inputs:
            analysis = self.analyze_single(
                text=item.text,
                rating=item.rating,
                title=item.title,
                date=item.date,
                verified_purchase=item.verified_purchase,
                review_id=item.id,
                check_duplicate=True
            )
            analyzed_reviews.append(analysis)
            all_aspect_mentions.extend(analysis.aspects)

        total_reviews = len(analyzed_reviews)
        spam_reviews = sum(1 for r in analyzed_reviews if r.spam.is_spam)
        valid_reviews = total_reviews - spam_reviews

        # 2. Aggregate Aspect Summaries
        aspect_summaries = self.aspect_scorer.aggregate(all_aspect_mentions)

        # 3. Overall Sentiment Stats (Filtering out spam for cleaner signal)
        usable_reviews = [r for r in analyzed_reviews if not r.spam.is_spam]
        if not usable_reviews:
            usable_reviews = analyzed_reviews

        pos_count = sum(1 for r in usable_reviews if r.sentiment.label == "POSITIVE")
        neg_count = sum(1 for r in usable_reviews if r.sentiment.label == "NEGATIVE")
        neu_count = sum(1 for r in usable_reviews if r.sentiment.label == "NEUTRAL")

        sentiment_dist = {"POSITIVE": pos_count, "NEUTRAL": neu_count, "NEGATIVE": neg_count}
        base_count = len(usable_reviews) or 1
        sentiment_pct = {
            "POSITIVE": round((pos_count / base_count) * 100, 1),
            "NEUTRAL": round((neu_count / base_count) * 100, 1),
            "NEGATIVE": round((neg_count / base_count) * 100, 1),
        }
        avg_compound = round(sum(r.sentiment.compound for r in usable_reviews) / base_count, 3)

        # 4. Top Positive & Negative Keywords via TF-IDF
        pos_texts = [r.cleaned_text for r in usable_reviews if r.sentiment.label == "POSITIVE"]
        neg_texts = [r.cleaned_text for r in usable_reviews if r.sentiment.label == "NEGATIVE"]
        top_pos_kw = self.keyword_extractor.extract_top_keywords(pos_texts, top_n=10)
        top_neg_kw = self.keyword_extractor.extract_top_keywords(neg_texts, top_n=10)

        # 5. Latent Topic Modeling (LDA)
        all_texts = [r.cleaned_text for r in usable_reviews]
        topics = self.topic_modeler.fit_transform(all_texts)

        # 6. Executive Insights & SWOT Analysis
        executive_summary = self.insights_engine.generate(analyzed_reviews, aspect_summaries)

        # 7. Timeline Series Analysis (if dates are present)
        timeline_data = self._aggregate_timeline(usable_reviews)

        return BatchAnalysisResult(
            total_reviews=total_reviews,
            valid_reviews=valid_reviews,
            spam_reviews=spam_reviews,
            overall_sentiment_distribution=sentiment_dist,
            overall_sentiment_percentages=sentiment_pct,
            average_compound_score=avg_compound,
            executive_summary=executive_summary,
            aspect_summaries=aspect_summaries,
            top_positive_keywords=top_pos_kw,
            top_negative_keywords=top_neg_kw,
            topics=topics,
            timeline=timeline_data,
            reviews=analyzed_reviews
        )

    def _aggregate_timeline(self, reviews: List[ReviewAnalysis]) -> Optional[List[Dict[str, Any]]]:
        """Aggregate sentiment trends over time."""
        date_reviews = [r for r in reviews if r.date]
        if len(date_reviews) < 3:
            return None

        try:
            df = pd.DataFrame([{"date": r.date, "compound": r.sentiment.compound} for r in date_reviews])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            if len(df) < 3:
                return None

            grouped = df.groupby(df["date"].dt.to_period("M")).agg(
                avg_sentiment=("compound", "mean"),
                review_count=("compound", "count")
            ).reset_index()

            grouped["date_str"] = grouped["date"].astype(str)
            return [
                {
                    "period": row["date_str"],
                    "average_sentiment": round(row["avg_sentiment"], 3),
                    "review_count": int(row["review_count"])
                }
                for _, row in grouped.iterrows()
            ]
        except Exception:
            return None

    @staticmethod
    def to_dataframe(batch_result: BatchAnalysisResult) -> pd.DataFrame:
        """Convert batch review results into an enriched Pandas DataFrame."""
        rows = []
        for r in batch_result.reviews:
            rows.append({
                "review_id": r.id,
                "text": r.original_text,
                "rating": r.rating,
                "predicted_rating": r.predicted_rating,
                "sentiment_label": r.sentiment.label,
                "sentiment_compound": r.sentiment.compound,
                "confidence": r.sentiment.confidence,
                "primary_emotion": r.sentiment.primary_emotion,
                "is_spam": r.spam.is_spam,
                "spam_score": r.spam.spam_score,
                "spam_reasons": "; ".join(r.spam.reasons),
                "aspects_mentioned": ", ".join(r.detected_aspects),
                "date": r.date,
                "verified_purchase": r.verified_purchase,
            })
        return pd.DataFrame(rows)
