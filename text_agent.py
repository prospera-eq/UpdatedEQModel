from typing import Dict, Any
from transformers import pipeline
import torch
import numpy as np

class TextAnalysisAgent:
    def __init__(self):
        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize emotion classification pipeline
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text communications for emotional intelligence and trust indicators.
        
        Args:
            text (str): The text content to analyze
            
        Returns:
            Dict containing analysis results including emotional scores and recommendations
        """
        try:
            # Get sentiment analysis
            sentiment_result = self.sentiment_analyzer(text)[0]
            sentiment_score = sentiment_result['score'] if sentiment_result['label'] == 'POSITIVE' else 1 - sentiment_result['score']
            
            # Get emotion classification
            emotion_result = self.emotion_classifier(text)[0]
            
            # Calculate trust score based on positive emotions
            trust_emotions = ['joy', 'love', 'optimism']
            trust_score = sentiment_score * (1.0 if emotion_result['label'] in trust_emotions else 0.7)
            
            # Generate recommendations based on analysis
            recommendations = [
                f"Emotional tone is {emotion_result['label']}",
                f"Consider {'maintaining' if sentiment_score > 0.6 else 'improving'} positive sentiment",
                f"Trust indicators are {'strong' if trust_score > 0.7 else 'moderate' if trust_score > 0.4 else 'weak'}"
            ]
            
            return {
                "emotional_score": float(sentiment_score),
                "trust_score": float(trust_score),
                "key_indicators": [
                    f"Primary emotion: {emotion_result['label']}",
                    f"Sentiment: {sentiment_result['label']}",
                    f"Confidence: {sentiment_result['score']:.2f}"
                ],
                "recommendations": recommendations
            }
            
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return {
                "error": str(e),
                "emotional_score": 0.5,
                "trust_score": 0.5,
                "key_indicators": [],
                "recommendations": ["Unable to analyze text"]
            }
