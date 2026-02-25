from typing import Dict, Any, List
from .base import BaseFeatureGenerator
from .generators import (
    SpamFeatureGenerator,
    AverageWordLengthFeatureGenerator,
    EmailEmbeddingsFeatureGenerator,
    RawEmailFeatureGenerator,
    NonTextCharacterFeatureGenerator,
)
from app.dataclasses import Email

# Constant list of available generators (NonTextCharacterFeatureGenerator added)
GENERATORS = {
    "spam": SpamFeatureGenerator,
    "word_length": AverageWordLengthFeatureGenerator,
    "email_embeddings": EmailEmbeddingsFeatureGenerator,
    "raw_email": RawEmailFeatureGenerator,
    "non_text_chars": NonTextCharacterFeatureGenerator,
}


class FeatureGeneratorFactory:
    """Factory for creating and managing feature generators"""
    
    def __init__(self):
        self._generators = GENERATORS
    
    def generate_all_features(self, email: Email, 
                            generator_names: List[str] = None) -> Dict[str, Any]:
        """Generate features using multiple generators"""
        if generator_names is None:
            generator_names = list(self._generators.keys())
        
        all_features = {}
        
        for gen_name in generator_names:
            generator_class = self._generators[gen_name]
            generator = generator_class()
            features = generator.generate_features(email)
            
            for feature_name, value in features.items():
                prefixed_name = f"{gen_name}_{feature_name}"
                all_features[prefixed_name] = value
        
        return all_features

    @classmethod
    def get_available_generators(cls) -> List[Dict[str, Any]]:
        """Return info about all available generators (for /features endpoint)"""
        result = []
        for name, gen_class in GENERATORS.items():
            gen = gen_class()
            result.append({
                "name": name,
                "features": gen.feature_names,
            })
        return result
