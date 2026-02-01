"""
IBM Cloudant client for incident persistence.
"""
import os
from datetime import datetime
from typing import Optional
from ibmcloudant.cloudant_v1 import CloudantV1, Document
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_cloud_sdk_core import ApiException


class CloudantClient:
    """Client for storing and retrieving incidents from IBM Cloudant."""

    DATABASE_NAME = "incidents"

    def __init__(self):
        self.client: Optional[CloudantV1] = None
        self.available = False
        self._initialize()

    def _initialize(self):
        """Initialize the Cloudant client."""
        api_key = os.getenv("CLOUDANT_API_KEY")
        url = os.getenv("CLOUDANT_URL")

        if not api_key or not url:
            print("Cloudant credentials not configured - using in-memory storage")
            return

        try:
            authenticator = IAMAuthenticator(api_key)
            self.client = CloudantV1(authenticator=authenticator)
            self.client.set_service_url(url)

            # Verify connection and ensure database exists
            self._ensure_database()
            self.available = True
            print("Cloudant client initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Cloudant: {e}")
            self.client = None

    def _ensure_database(self):
        """Create the incidents database if it doesn't exist."""
        try:
            self.client.get_database_information(db=self.DATABASE_NAME)
            print(f"Database '{self.DATABASE_NAME}' exists")
        except ApiException as e:
            if e.code == 404:
                print(f"Creating database '{self.DATABASE_NAME}'...")
                self.client.put_database(db=self.DATABASE_NAME)
                print(f"Database '{self.DATABASE_NAME}' created")
            else:
                raise

    def save_incident(self, incident: dict) -> str:
        """
        Save an incident to Cloudant.

        Args:
            incident: The incident data to save

        Returns:
            The document ID
        """
        if not self.available:
            return incident.get("incident_id", "")

        try:
            # Use incident_id as document _id
            doc_id = incident.get("incident_id", "")

            # Prepare document
            doc = Document(
                id=doc_id,
                **{k: v for k, v in incident.items() if k != "incident_id"}
            )
            doc._id = doc_id

            response = self.client.post_document(
                db=self.DATABASE_NAME,
                document=incident
            ).get_result()

            print(f"Incident saved to Cloudant: {response.get('id')}")
            return response.get("id", doc_id)
        except Exception as e:
            print(f"Failed to save incident to Cloudant: {e}")
            return incident.get("incident_id", "")

    def get_incident(self, incident_id: str) -> Optional[dict]:
        """
        Retrieve an incident by ID.

        Args:
            incident_id: The incident ID to retrieve

        Returns:
            The incident data or None if not found
        """
        if not self.available:
            return None

        try:
            # Search by incident_id field
            selector = {"incident_id": {"$eq": incident_id}}
            response = self.client.post_find(
                db=self.DATABASE_NAME,
                selector=selector,
                limit=1
            ).get_result()

            docs = response.get("docs", [])
            if docs:
                doc = docs[0]
                # Remove Cloudant metadata
                doc.pop("_id", None)
                doc.pop("_rev", None)
                return doc
            return None
        except Exception as e:
            print(f"Failed to get incident from Cloudant: {e}")
            return None

    def list_incidents(self, limit: int = 50) -> list:
        """
        List recent incidents.

        Args:
            limit: Maximum number of incidents to return

        Returns:
            List of incident summaries
        """
        if not self.available:
            return []

        try:
            # Get all documents with basic info
            response = self.client.post_all_docs(
                db=self.DATABASE_NAME,
                include_docs=True,
                limit=limit
            ).get_result()

            incidents = []
            for row in response.get("rows", []):
                doc = row.get("doc", {})
                if doc and not doc.get("_id", "").startswith("_"):
                    incidents.append({
                        "id": doc.get("incident_id", doc.get("_id")),
                        "incident_type": doc.get("incident_type", "unknown"),
                        "timestamp": doc.get("timestamp", ""),
                        "severity": doc.get("severity", "unknown")
                    })

            return incidents
        except Exception as e:
            print(f"Failed to list incidents from Cloudant: {e}")
            return []

    def search_similar_incidents(self, error_patterns: list, limit: int = 5) -> list:
        """
        Search for similar incidents based on error patterns.

        Args:
            error_patterns: List of error patterns to match
            limit: Maximum number of results

        Returns:
            List of similar incidents
        """
        if not self.available or not error_patterns:
            return []

        try:
            # Search for incidents with matching error patterns
            selector = {
                "nlu_analysis.error_patterns": {
                    "$elemMatch": {"$in": error_patterns}
                }
            }

            response = self.client.post_find(
                db=self.DATABASE_NAME,
                selector=selector,
                limit=limit
            ).get_result()

            similar = []
            for doc in response.get("docs", []):
                similar.append({
                    "id": doc.get("incident_id", doc.get("_id")),
                    "severity": doc.get("severity", "unknown"),
                    "created_at": doc.get("timestamp", ""),
                    "matching_pattern": next(
                        (p for p in error_patterns
                         if p in doc.get("nlu_analysis", {}).get("error_patterns", [])),
                        error_patterns[0] if error_patterns else ""
                    )
                })

            return similar
        except Exception as e:
            print(f"Failed to search similar incidents: {e}")
            return []

    def health_check(self) -> str:
        """Check if Cloudant is available."""
        if not self.available:
            return "not_configured"

        try:
            self.client.get_database_information(db=self.DATABASE_NAME)
            return "active"
        except Exception:
            return "error"


# Singleton instance
_cloudant_client: Optional[CloudantClient] = None


def get_cloudant_client() -> CloudantClient:
    """Get or create the Cloudant client singleton."""
    global _cloudant_client
    if _cloudant_client is None:
        _cloudant_client = CloudantClient()
    return _cloudant_client
