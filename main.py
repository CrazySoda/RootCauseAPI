"""
Root Cause Analysis API Service
Integrates IBM NLU, watsonx.ai Granite, and GitHub API for intelligent incident analysis.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime
import uuid
import os

load_dotenv()

from nlu_analyzer import NLUAnalyzer
from watsonx_client import WatsonxClient
from github_client import GitHubClient
from cloudant_client import get_cloudant_client, CloudantClient

app = FastAPI(
    title="Root Cause Analysis API",
    description="Intelligent incident root cause analysis using IBM AI services",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
nlu_analyzer = None
watsonx_client = None
github_client = None
cloudant_client: CloudantClient = None

# In-memory incident storage (fallback when Cloudant not available)
incidents_store: List[Dict] = []


@app.on_event("startup")
async def startup():
    global nlu_analyzer, watsonx_client, github_client, cloudant_client

    try:
        nlu_analyzer = NLUAnalyzer()
        print("NLU Analyzer initialized")
    except Exception as e:
        print(f"Warning: NLU not available: {e}")

    try:
        watsonx_client = WatsonxClient()
        print("Watsonx Client initialized")
    except Exception as e:
        print(f"Warning: Watsonx not available: {e}")

    # GitHub client doesn't require env vars - accepts token per-request
    github_client = GitHubClient()
    print("GitHub Client initialized (accepts PAT per-request)")

    try:
        cloudant_client = get_cloudant_client()
        if cloudant_client.available:
            print("Cloudant Client initialized")
        else:
            print("Cloudant not configured - using in-memory storage")
    except Exception as e:
        print(f"Warning: Cloudant not available: {e}")


# ============ Request/Response Models ============

class AnalyzeRequest(BaseModel):
    error_log: str
    repo_url: Optional[str] = None
    github_pat: Optional[str] = None  # User provides their GitHub PAT
    incident_type: str = "unknown"


class AnalyzeResponse(BaseModel):
    incident_id: str
    severity: str
    root_cause: str
    evidence: List[str]
    suggested_fix: str
    confidence: float
    similar_incidents: List[Dict]
    nlu_analysis: Dict


class QuickAnalyzeRequest(BaseModel):
    error_log: str


# ============ Endpoints ============

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_root_cause(request: AnalyzeRequest):
    """
    Full root cause analysis endpoint.

    Steps:
    1. Use IBM NLU to extract entities and keywords from error log
    2. Search GitHub for related code (if repo_url provided)
    3. Use IBM watsonx.ai (Granite) to generate root cause analysis
    4. Store incident for history
    5. Find similar past incidents
    """

    # Step 1: NLU Analysis
    if nlu_analyzer:
        nlu_result = nlu_analyzer.analyze_error_log(request.error_log)
    else:
        # Fallback: Basic keyword extraction
        words = request.error_log.split()
        nlu_result = {
            "keywords": [{"text": word} for word in words[:10]],
            "error_patterns": [],
            "categories": [],
            "entities": []
        }
        # Extract basic error patterns
        for pattern in ["Exception", "Error", "Failed", "Timeout", "500", "404"]:
            if pattern.lower() in request.error_log.lower():
                nlu_result["error_patterns"].append(pattern)

    # Step 2: Get code context from GitHub (if repo and PAT provided)
    code_context = ""
    if request.repo_url and request.github_pat and github_client:
        try:
            error_patterns = nlu_result.get("error_patterns", [])
            if error_patterns:
                code_context = await github_client.search_and_get_context(
                    request.repo_url,
                    error_patterns,
                    request.github_pat  # Pass user's PAT
                )
        except Exception as e:
            print(f"GitHub search failed: {e}")

    # Step 3: Generate root cause analysis with watsonx.ai
    analysis_text = ""
    severity = "Medium"
    confidence = 0.5

    if watsonx_client:
        analysis_result = watsonx_client.analyze_root_cause(
            error_info=nlu_result,
            code_context=code_context,
            incident_type=request.incident_type
        )
        analysis_text = analysis_result.get("analysis", "")
        severity = analysis_result.get("severity", "Medium")
        confidence = analysis_result.get("confidence", 0.5)
    else:
        # Fallback analysis without LLM
        patterns = nlu_result.get('error_patterns', [])
        keywords = [k['text'] for k in nlu_result.get('keywords', [])[:5]]
        analysis_text = f"""
Probable Root Cause: Based on error patterns {patterns}, there appears to be an issue in the application.

Evidence: Keywords found - {keywords}

Suggested Fix: Review the code related to the error patterns and add proper error handling.

Severity: Medium
Confidence: 50%
"""

    # Step 4: Store incident
    incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.utcnow().isoformat()
    incident_data = {
        "incident_id": incident_id,
        "timestamp": timestamp,
        "error_log": request.error_log[:500],
        "repo_url": request.repo_url,
        "incident_type": request.incident_type,
        "severity": severity,
        "confidence": confidence,
        "nlu_analysis": nlu_result,
        "root_cause_analysis": analysis_text
    }

    # Save to Cloudant if available, otherwise use in-memory
    if cloudant_client and cloudant_client.available:
        cloudant_client.save_incident(incident_data)
    else:
        incidents_store.append(incident_data)

    # Step 5: Find similar incidents
    error_patterns = nlu_result.get("error_patterns", [])
    if cloudant_client and cloudant_client.available:
        similar = cloudant_client.search_similar_incidents(error_patterns)
    else:
        similar = _find_similar_incidents(error_patterns)

    return AnalyzeResponse(
        incident_id=incident_id,
        severity=severity,
        root_cause=analysis_text,
        evidence=nlu_result.get("error_patterns", []),
        suggested_fix="See analysis above for specific recommendations",
        confidence=confidence,
        similar_incidents=similar,
        nlu_analysis=nlu_result
    )


@app.post("/analyze-log-only")
async def analyze_log_only(request: QuickAnalyzeRequest):
    """
    Quick analysis using only IBM NLU (no LLM call).
    Fast endpoint for initial triage.
    """
    if not nlu_analyzer:
        # Fallback without NLU
        words = request.error_log.split()
        patterns = []
        for p in ["Exception", "Error", "Failed", "Timeout", "500", "404"]:
            if p.lower() in request.error_log.lower():
                patterns.append(p)

        return {
            "quick_summary": {
                "error_patterns": patterns,
                "top_keywords": words[:5],
                "categories": []
            },
            "message": "NLU service not available, using basic analysis"
        }

    result = nlu_analyzer.analyze_error_log(request.error_log)

    return {
        "analysis": result,
        "quick_summary": {
            "error_patterns": result.get("error_patterns", []),
            "top_keywords": [k["text"] for k in result.get("keywords", [])[:5]],
            "categories": [c.get("label") for c in result.get("categories", [])]
        }
    }


@app.get("/incidents/{incident_id}")
async def get_incident(incident_id: str):
    """Retrieve a stored incident by ID."""
    # Try Cloudant first
    if cloudant_client and cloudant_client.available:
        incident = cloudant_client.get_incident(incident_id)
        if incident:
            return incident

    # Fallback to in-memory
    for incident in incidents_store:
        if incident.get("incident_id") == incident_id:
            return incident

    raise HTTPException(status_code=404, detail="Incident not found")


@app.get("/incidents")
async def list_incidents(limit: int = 50):
    """List recent incidents."""
    # Try Cloudant first
    if cloudant_client and cloudant_client.available:
        incidents = cloudant_client.list_incidents(limit=limit)
        return {
            "incidents": incidents,
            "total": len(incidents),
            "storage": "cloudant"
        }

    # Fallback to in-memory
    return {
        "incidents": incidents_store[-limit:] if incidents_store else [],
        "total": len(incidents_store),
        "storage": "in-memory"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    cloudant_status = "inactive"
    if cloudant_client:
        cloudant_status = cloudant_client.health_check()

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "nlu": "active" if nlu_analyzer else "inactive",
            "watsonx": "active" if watsonx_client else "inactive",
            "github": "available" if github_client else "inactive",  # Accepts PAT per-request
            "cloudant": cloudant_status
        }
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Root Cause Analysis API",
        "version": "1.0.0",
        "description": "Intelligent incident root cause analysis using IBM AI services",
        "endpoints": {
            "POST /analyze": "Full root cause analysis",
            "POST /analyze-log-only": "Quick NLU-only triage",
            "GET /incidents": "List stored incidents",
            "GET /incidents/{id}": "Get specific incident",
            "GET /health": "Health check"
        }
    }


def _find_similar_incidents(error_patterns: List[str]) -> List[Dict]:
    """Find similar past incidents based on error patterns (in-memory fallback)."""
    similar = []
    for incident in incidents_store[-20:]:  # Check last 20 incidents
        incident_patterns = incident.get("nlu_analysis", {}).get("error_patterns", [])
        # Check for pattern overlap
        for pattern in error_patterns:
            if pattern in incident_patterns:
                similar.append({
                    "id": incident.get("incident_id"),
                    "severity": incident.get("severity"),
                    "created_at": incident.get("timestamp"),
                    "matching_pattern": pattern
                })
                break
    return similar[:5]  # Return top 5 similar


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
