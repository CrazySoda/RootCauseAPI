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

            # Extract code references (files, classes, methods)
            code_refs = self.extract_code_references(log_text)

            return {
                "entities": self._process_entities(response.get("entities", [])),
                "keywords": self._process_keywords(response.get("keywords", [])),
                "categories": response.get("categories", []),
                "error_patterns": self._extract_error_patterns(log_text),
                "code_references": code_refs,
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

    def extract_code_references(self, log_text: str) -> Dict[str, List[str]]:
        """
        Extract specific code references from error logs/stack traces.
        Returns file names, class names, method names, and line numbers.
        """
        import re

        references = {
            "files": [],
            "classes": [],
            "methods": [],
            "search_terms": []  # Combined terms for GitHub search
        }

        # Pattern for Java stack traces: at com.package.Class.method(File.java:123)
        java_pattern = r'at\s+([\w\.]+)\.([\w]+)\(([\w]+\.java):(\d+)\)'
        for match in re.finditer(java_pattern, log_text):
            class_path = match.group(1)
            method = match.group(2)
            file = match.group(3)
            line = match.group(4)

            # Extract class name from full path
            class_name = class_path.split('.')[-1]

            if file not in references["files"]:
                references["files"].append(file)
            if class_name not in references["classes"]:
                references["classes"].append(class_name)
            if method not in references["methods"]:
                references["methods"].append(method)

            # Add specific search term
            references["search_terms"].append(f"{class_name} {method}")

        # Pattern for Python stack traces: File "path/to/file.py", line 123, in method_name
        python_pattern = r'File\s+"([^"]+\.py)",\s+line\s+(\d+),\s+in\s+(\w+)'
        for match in re.finditer(python_pattern, log_text):
            file_path = match.group(1)
            line = match.group(2)
            method = match.group(3)

            # Extract just filename
            file = file_path.split('/')[-1].split('\\')[-1]

            if file not in references["files"]:
                references["files"].append(file)
            if method not in references["methods"] and method != "<module>":
                references["methods"].append(method)

            references["search_terms"].append(f"{file} {method}")

        # Pattern for JavaScript/Node stack traces: at methodName (path/file.js:123:45)
        js_pattern = r'at\s+(\w+)?\s*\(?([^\s\)]+\.(js|ts|tsx)):(\d+):\d+\)?'
        for match in re.finditer(js_pattern, log_text):
            method = match.group(1) if match.group(1) else ""
            file_path = match.group(2)

            file = file_path.split('/')[-1].split('\\')[-1]

            if file not in references["files"]:
                references["files"].append(file)
            if method and method not in references["methods"]:
                references["methods"].append(method)

            if method:
                references["search_terms"].append(f"{file} {method}")
            else:
                references["search_terms"].append(file)

        # Pattern for generic "ClassName.methodName" references
        class_method_pattern = r'\b([A-Z][a-zA-Z0-9]+)\.([a-z][a-zA-Z0-9]+)\b'
        for match in re.finditer(class_method_pattern, log_text):
            class_name = match.group(1)
            method = match.group(2)

            if class_name not in references["classes"]:
                references["classes"].append(class_name)
            if method not in references["methods"]:
                references["methods"].append(method)

            references["search_terms"].append(f"{class_name} {method}")

        # Pattern for file references: filename.ext:line
        file_line_pattern = r'\b([\w\-]+\.(java|py|js|ts|tsx|go|rb|cpp|c|rs)):(\d+)\b'
        for match in re.finditer(file_line_pattern, log_text):
            file = match.group(1)
            if file not in references["files"]:
                references["files"].append(file)
                references["search_terms"].append(file)

        # Remove duplicates and limit
        references["search_terms"] = list(dict.fromkeys(references["search_terms"]))[:10]
        references["files"] = list(dict.fromkeys(references["files"]))[:10]
        references["classes"] = list(dict.fromkeys(references["classes"]))[:10]
        references["methods"] = list(dict.fromkeys(references["methods"]))[:10]

        return references
