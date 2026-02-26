from typing import Dict, Any
from app.models.similarity_model import EmailClassifierModel
from app.features.factory import FeatureGeneratorFactory
from app.dataclasses import Email

class EmailTopicInferenceService:
    """Service that orchestrates email topic classification using feature similarity matching"""
    
    def __init__(self):
        self.model = EmailClassifierModel()
        self.feature_factory = FeatureGeneratorFactory()
    
    def classify_email(self, email: Email, method: str = "topic") -> Dict[str, Any]:
        """Classify an email into topics using generated features"""
        
        # Step 1: Generate features from email
        features = self.feature_factory.generate_all_features(email)
        
        if method == "email":
            # Email-similarity classification
            email_result = self.model.predict_by_email_similarity(features)
            return {
                "classification_method": "email_similarity",
                "predicted_topic": email_result["predicted_topic"],
                "similarity": email_result.get("similarity", 0.0),
                "best_match": email_result.get("best_match"),
                "top_email_scores": email_result.get("email_scores", []),
                "features": features,
                "available_topics": self.model.topics,
            }
        else:
            # Default: topic-description classification
            predicted_topic = self.model.predict(features)
            topic_scores = self.model.get_topic_scores(features)
            return {
                "classification_method": "topic_similarity",
                "predicted_topic": predicted_topic,
                "topic_scores": topic_scores,
                "features": features,
                "available_topics": self.model.topics,
            }
    
    def reload_topics(self):
        """Reload topics from disk after adding new ones"""
        self.model.reload_topics()
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        return {
            "available_topics": self.model.topics,
            "topics_with_descriptions": self.model.get_all_topics_with_descriptions(),
        }