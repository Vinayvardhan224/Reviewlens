"""
Pydantic data models and schemas for ReviewLens.
Provides structured types for single reviews, aspect mentions, spam flags, and aggregate metrics.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class ReviewInput(BaseModel):
    """Input representation of a product review."""
    id: Optional[str] = None
    text: str
    title: Optional[str] = None
    rating: Optional[float] = Field(None, ge=1.0, le=5.0, description="Star rating between 1 and 5")
    author: Optional[str] = None
    date: Optional[str] = None
    verified_purchase: Optional[bool] = None
    product_id: Optional[str] = None
    category: Optional[str] = None


class SpamResult(BaseModel):
    """Spam and fake review detection output."""
    is_spam: bool = False
    spam_score: float = Field(0.0, ge=0.0, le=1.0, description="Probability or heuristic spam index")
    reasons: List[str] = Field(default_factory=list)


class SentimentResult(BaseModel):
    """Sentiment classification and polarity metrics."""
    label: str = Field(..., description="POSITIVE, NEUTRAL, or NEGATIVE")
    compound: float = Field(..., ge=-1.0, le=1.0, description="Normalized polarity score [-1.0, 1.0]")
    pos: float = Field(..., ge=0.0, le=1.0)
    neu: float = Field(..., ge=0.0, le=1.0)
    neg: float = Field(..., ge=0.0, le=1.0)
    predicted_rating: float = Field(..., ge=1.0, le=5.0, description="Predicted 1-5 star rating")
    confidence: float = Field(..., ge=0.0, le=1.0)
    primary_emotion: Optional[str] = None
    emotion_scores: Dict[str, float] = Field(default_factory=dict)


class AspectMention(BaseModel):
    """An individual aspect mention extracted from a review sentence."""
    category: str
    matched_term: str
    sentence: str
    sentiment_label: str
    sentiment_score: float
    opinion_words: List[str] = Field(default_factory=list)


class ReviewAnalysis(BaseModel):
    """Fully analyzed product review with sentiment, spam, aspects, and emotions."""
    id: Optional[str] = None
    original_text: str
    cleaned_text: str
    rating: Optional[float] = None
    predicted_rating: float
    sentiment: SentimentResult
    spam: SpamResult
    aspects: List[AspectMention] = Field(default_factory=list)
    detected_aspects: List[str] = Field(default_factory=list)
    date: Optional[str] = None
    verified_purchase: Optional[bool] = None


class AspectSummary(BaseModel):
    """Aggregated statistics for a specific product aspect across reviews."""
    category: str
    total_mentions: int
    positive_mentions: int
    negative_mentions: int
    neutral_mentions: int
    average_sentiment: float
    net_sentiment_score: float = Field(..., description="% Positive - % Negative, range [-100, 100]")
    sentiment_ratio: Dict[str, float] = Field(default_factory=dict)
    sample_positive_quotes: List[str] = Field(default_factory=list)
    sample_negative_quotes: List[str] = Field(default_factory=list)


class TopicCluster(BaseModel):
    """Topic cluster discovered via unsupervised topic modeling."""
    topic_id: int
    label: str
    keywords: List[str]
    sample_reviews: List[str] = Field(default_factory=list)


class ExecutiveSummary(BaseModel):
    """Executive insights, SWOT breakdown, and strategic recommendations."""
    net_sentiment_score: float = Field(..., description="Overall Net Sentiment Score [-100, 100]")
    csat_score: float = Field(..., description="Estimated Customer Satisfaction Index [0, 100]%")
    sentiment_distribution: Dict[str, int]
    sentiment_percentages: Dict[str, float]
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    threats: List[str] = Field(default_factory=list)
    actionable_recommendations: List[Dict[str, Any]] = Field(default_factory=list)


class BatchAnalysisResult(BaseModel):
    """Comprehensive result object returned after batch review analysis."""
    total_reviews: int
    valid_reviews: int
    spam_reviews: int
    overall_sentiment_distribution: Dict[str, int]
    overall_sentiment_percentages: Dict[str, float]
    average_compound_score: float
    executive_summary: ExecutiveSummary
    aspect_summaries: List[AspectSummary]
    top_positive_keywords: List[Dict[str, Any]]
    top_negative_keywords: List[Dict[str, Any]]
    topics: List[TopicCluster]
    timeline: Optional[List[Dict[str, Any]]] = None
    reviews: List[ReviewAnalysis]
