import numpy as np
from transformers import pipeline
from keybert import KeyBERT
import yake
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from typing import List, Dict, Any
from datetime import datetime

class AIEngine:
    def __init__(self):
        # Initialize models (in production, use singleton or lazy loading)
        self.sentiment_model = pipeline(
            "sentiment-analysis", 
            model="cardiffnlp/twitter-roberta-base-sentiment",
            device=-1 # Use CPU by default, change to 0 for GPU
        )
        self.kw_model = KeyBERT()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.topic_model = BERTopic(embedding_model=self.embedding_model)

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a single text."""
        result = self.sentiment_model(text[:512])[0]
        # Map labels (0 -> Negative, 1 -> Neutral, 2 -> Positive)
        label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
        return {
            "label": label_map.get(result["label"], result["label"]),
            "score": float(result["score"])
        }

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract keywords using KeyBERT and YAKE."""
        # KeyBERT for semantic keywords
        keywords = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
        return [kw[0] for kw in keywords]

    def cluster_topics(self, docs: List[str]) -> List[int]:
        """Cluster documents into topics."""
        topics, _ = self.topic_model.fit_transform(docs)
        return topics

    def calculate_trend_score(self, frequency: int, velocity: float, recency: float) -> float:
        """
        Calculate a trend score based on volume, speed, and time.
        score = (frequency * velocity * recency_weight)
        """
        return float(frequency * velocity * recency)

ai_engine = AIEngine()
