import os
import json
import numpy as np
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer

class EmailClassifierModel:
    """Email classifier model using embedding similarity"""

    def __init__(self):
        self.topic_data = self._load_topic_data()
        self.topics = list(self.topic_data.keys())

        # Load sentence transformer model (same model as feature generator)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Pre-compute embeddings for all topic descriptions
        self.topic_embeddings = self._compute_topic_embeddings()
    
    def _load_topic_data(self) -> Dict[str, Dict[str, Any]]:
        """Load topic data from data/topic_keywords.json"""
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'topic_keywords.json')
        with open(data_file, 'r') as f:
            return json.load(f)
            
    def _load_stored_emails(self) -> List[Dict[str, Any]]:
        """Load stored emails from data/emails.json"""
        data_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'emails.json',
        )
        with open(data_file, 'r') as f:
            return json.load(f)

    def _compute_topic_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for all topic descriptions"""
        topic_embeddings = {}
        for topic, data in self.topic_data.items():
            description = data['description']
            embedding = self.model.encode(description, convert_to_numpy=True)
            topic_embeddings[topic] = embedding
        return topic_embeddings

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))
    
    def predict(self, features: Dict[str, Any]) -> str:
        """Classify email into one of the topics using feature similarity"""
        scores = {}
        
        # Calculate similarity scores for each topic based on features
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = score
        
        return max(scores, key=scores.get)
    
    def get_topic_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get classification scores for all topics"""
        scores = {}
        
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = float(score)
        
        return scores
    
    def _calculate_topic_score(self, features: Dict[str, Any], topic: str) -> float:
        """Cosine similarity between email embedding and topic embedding"""
        email_embedding = features.get("email_embeddings_average_embedding", None)
        if email_embedding is None:
            return 0.0
        if isinstance(email_embedding, list):
            email_embedding = np.array(email_embedding)
        topic_embedding = self.topic_embeddings[topic]

        sim = self._cosine_similarity(email_embedding, topic_embedding)
        return (sim + 1) / 2  # normalize to 0-1

    #  Email-based classification

    def predict_by_email_similarity(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Classify email by finding the most similar stored email.
        
        Returns dict with predicted_topic, similarity, best_match info, and all scores.
        """
        email_embedding = features.get("email_embeddings_average_embedding", None)
        if email_embedding is None:
            return {
                "predicted_topic": "unknown",
                "similarity": 0.0,
                "best_match": None,
                "email_scores": [],
            }
        if isinstance(email_embedding, list):
            email_embedding = np.array(email_embedding)

        stored_emails = self._load_stored_emails()

        # Filter to only emails that have a ground_truth label
        labeled_emails = [e for e in stored_emails if e.get("ground_truth")]
        if not labeled_emails:
            return {
                "predicted_topic": "unknown",
                "similarity": 0.0,
                "best_match": None,
                "email_scores": [],
                "message": "No labeled emails stored yet. Add emails with ground_truth first.",
            }

        # Compute similarity to every stored email
        scores = []
        for stored in labeled_emails:
            stored_text = f"{stored['subject']} {stored['body']}"
            stored_emb = self.model.encode(stored_text, convert_to_numpy=True)
            sim = self._cosine_similarity(email_embedding, stored_emb)
            scores.append({
                "email_id": stored.get("id"),
                "subject": stored["subject"],
                "ground_truth": stored["ground_truth"],
                "similarity": float(sim),
            })

        scores.sort(key=lambda x: x["similarity"], reverse=True)
        best = scores[0]

        return {
            "predicted_topic": best["ground_truth"],
            "similarity": best["similarity"],
            "best_match": best,
            "email_scores": scores[:5],  # top 5
        }

    #  Helpers

    def reload_topics(self):
        """Reload topics from disk (called after adding a new topic)"""
        self.topic_data = self._load_topic_data()
        self.topics = list(self.topic_data.keys())
        self.topic_embeddings = self._compute_topic_embeddings()

    def get_topic_description(self, topic: str) -> str:
        return self.topic_data[topic]['description']

    def get_all_topics_with_descriptions(self) -> Dict[str, str]:
        return {topic: self.get_topic_description(topic) for topic in self.topics}