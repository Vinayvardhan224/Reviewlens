"""
ReviewLens Aspect-Based Sentiment Analysis (ABSA) Package.
"""

from reviewlens.absa.aspect_categories import AspectTaxonomy
from reviewlens.absa.aspect_extractor import AspectExtractor
from reviewlens.absa.aspect_scorer import AspectScorer

__all__ = ["AspectTaxonomy", "AspectExtractor", "AspectScorer"]
