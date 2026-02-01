"""
IBM Natural Language Understanding integration for log analysis.
Extracts entities, keywords, and categories from error logs.
"""

import os
from typing import Dict, Any, List
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, CategoriesOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


class NLUAnalyzer:
    def __init__(self):
        api_key = os.getenv("NLU_API_KEY")
        url = os.getenv("NLU_URL")

        if not api_key or not url:
            raise ValueError("NLU_API_KEY and NLU_URL must be set")

        authenticator = IAMAuthenticator(api_key)
        self.nlu = NaturalLanguageUnderstandingV1(
            version='2022-04-07',
            authenticator=authenticator
        )
        self.nlu.set_service_url(url)

    def analyze_error_log(self, log_text: str) -> Dict[str, Any]:
        """
        Analyze error log text to extract:
        - Error types (entities)
        - Key terms (keywords)
        - Categories (error category)
        """
        try:
            response = self.nlu.analyze(
                text=log_text,
                features=Features(
                    entities=EntitiesOptions(
                        sentiment=True,
                        limit=10
                    ),
                    keywords=KeywordsOptions(
                        sentiment=True,
                        emotion=True,
                        limit=15
                    ),
                    categories=CategoriesOptions(
                        limit=3
                    )
                )
            ).get_result()

            return {
                "entities": self._process_entities(response.get("entities", [])),
                "keywords": self._process_keywords(response.get("keywords", [])),
                "categories": response.get("categories", []),
                "error_patterns": self._extract_error_patterns(log_text),
                "raw_analysis": response
            }
        except Exception as e:
            return {"error": str(e)}

    def _process_entities(self, entities: List) -> List[Dict]:
        """Process NLU entities into structured format."""
        return [
            {
                "type": e.get("type"),
                "text": e.get("text"),
                "relevance": e.get("relevance"),
                "confidence": e.get("confidence")
            }
            for e in entities
        ]

    def _process_keywords(self, keywords: List) -> List[Dict]:
        """Process NLU keywords into structured format."""
        return [
            {
                "text": k.get("text"),
                "relevance": k.get("relevance"),
                "sentiment": k.get("sentiment", {}).get("label") if k.get("sentiment") else None
            }
            for k in keywords
        ]

    def _extract_error_patterns(self, log_text: str) -> List[str]:
        """Extract common error patterns from log text."""
        patterns = []
        error_indicators = [
            "Exception", "Error", "Failed", "Timeout",
            "NullPointer", "OutOfMemory", "Connection refused",
            "404", "500", "503", "FATAL", "CRITICAL",
            "TypeError", "ValueError", "KeyError", "AttributeError",
            "IndexError", "ImportError", "ModuleNotFoundError"
        ]

        for indicator in error_indicators:
            if indicator.lower() in log_text.lower():
                patterns.append(indicator)

        return patterns
