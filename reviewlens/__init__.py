"""
ReviewLens: An NLP-Based Smart Product Review Analysis System.
Author: Google DeepMind Pair Programming
"""

__version__ = "1.0.0"

from reviewlens.pipeline import ReviewLensPipeline
from reviewlens.models import (
    ReviewInput,
    ReviewAnalysis,
    BatchAnalysisResult,
    AspectSummary,
    AspectMention,
    SentimentResult,
    SpamResult,
    ExecutiveSummary,
    TopicCluster,
)
from reviewlens.sentiment.vader_analyzer import SentimentAnalyzer
from reviewlens.sentiment.emotion_analyzer import EmotionAnalyzer
from reviewlens.absa.aspect_extractor import AspectExtractor
from reviewlens.absa.aspect_categories import AspectTaxonomy
from reviewlens.spam_detector.detector import SpamDetector

__all__ = [
    "ReviewLensPipeline",
    "ReviewInput",
    "ReviewAnalysis",
    "BatchAnalysisResult",
    "AspectSummary",
    "AspectMention",
    "SentimentResult",
    "SpamResult",
    "ExecutiveSummary",
    "TopicCluster",
    "SentimentAnalyzer",
    "EmotionAnalyzer",
    "AspectExtractor",
    "AspectTaxonomy",
    "SpamDetector",
]
