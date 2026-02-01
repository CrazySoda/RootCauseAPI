"""
IBM watsonx.ai integration for intelligent code analysis.
Uses IBM Granite models for root cause analysis.
"""

import os
from typing import Dict, Any
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference


class WatsonxClient:
    def __init__(self):
        api_key = os.getenv("WATSONX_API_KEY")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

        if not api_key or not project_id:
            raise ValueError("WATSONX_API_KEY and WATSONX_PROJECT_ID must be set")

        self.credentials = Credentials(
            url=url,
            api_key=api_key
        )
        self.project_id = project_id

        # Use Granite 3.0 8B Instruct (recommended for hackathon)
        self.model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            credentials=self.credentials,
            project_id=project_id,
            params={
                "decoding_method": "greedy",
                "max_new_tokens": 1024,
                "min_new_tokens": 0,
                "repetition_penalty": 1.0
            }
        )

    def analyze_root_cause(
        self,
        error_info: Dict[str, Any],
        code_context: str,
        incident_type: str
    ) -> Dict[str, Any]:
        """
        Generate root cause analysis using Granite model.
        """
        keywords = [k.get('text', '') for k in error_info.get('keywords', [])]
        error_patterns = error_info.get('error_patterns', [])
        categories = [c.get('label', '') for c in error_info.get('categories', [])]

        prompt = f"""<|system|>
You are a senior DevOps engineer and root cause analyst. Analyze the following incident and provide a detailed root cause analysis.

<|user|>
## Incident Type: {incident_type}

## Error Information:
- Keywords: {', '.join(keywords[:10])}
- Error Patterns: {', '.join(error_patterns)}
- Categories: {', '.join(categories)}

## Related Code Context:
{code_context[:3000] if code_context else 'No code context available'}

Please provide:
1. **Probable Root Cause** (1-2 sentences)
2. **Evidence** (list specific code/log indicators)
3. **Suggested Fix** (actionable steps)
4. **Severity Assessment** (Critical/High/Medium/Low)
5. **Confidence Score** (0-100%)

<|assistant|>
"""

        try:
            response = self.model.generate(prompt)
            generated_text = response.get("results", [{}])[0].get("generated_text", "")

            # Parse severity from response
            severity = "Medium"
            confidence = 0.5
            if "Critical" in generated_text:
                severity = "Critical"
                confidence = 0.85
            elif "High" in generated_text:
                severity = "High"
                confidence = 0.75
            elif "Low" in generated_text:
                severity = "Low"
                confidence = 0.6

            return {
                "analysis": generated_text,
                "severity": severity,
                "confidence": confidence,
                "model_used": "ibm/granite-3-8b-instruct",
                "status": "success"
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed",
                "severity": "Unknown",
                "confidence": 0.0
            }

    def suggest_fix(self, root_cause: str, code_snippet: str) -> str:
        """
        Generate fix suggestion for identified root cause.
        """
        prompt = f"""<|system|>
You are a helpful coding assistant. Suggest a fix for the following issue.

<|user|>
## Root Cause:
{root_cause}

## Problematic Code:
```
{code_snippet[:2000]}
```

Provide a specific code fix or configuration change.

<|assistant|>
"""

        try:
            response = self.model.generate(prompt)
            return response.get("results", [{}])[0].get("generated_text", "Unable to generate fix")
        except Exception as e:
            return f"Error generating fix: {str(e)}"
