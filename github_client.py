"""
GitHub API integration for code search and retrieval.
Uses GitHub REST API with PAT token - no local cloning required.
Token is provided per-request by the user.
"""

import httpx
from typing import Dict, Any, List, Optional
import base64


class GitHubClient:
    """GitHub client that accepts token per-method call."""

    def __init__(self):
        self.base_url = "https://api.github.com"

    def _get_headers(self, token: str) -> dict:
        """Generate headers with the provided token."""
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }

    def _parse_repo_url(self, repo_url: str) -> tuple:
        """Parse owner and repo name from GitHub URL."""
        # Handle various URL formats
        url = repo_url.replace("https://github.com/", "").replace("http://github.com/", "")
        url = url.rstrip("/").replace(".git", "")
        parts = url.split("/")
        if len(parts) >= 2:
            return parts[0], parts[1]
        raise ValueError(f"Invalid GitHub URL: {repo_url}")

    async def search_code(
        self,
        repo_url: str,
        pattern: str,
        token: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search for code patterns in a repository using GitHub Code Search API.

        Args:
            repo_url: GitHub repository URL
            pattern: Search pattern
            token: GitHub Personal Access Token
            max_results: Maximum number of results to return
        """
        owner, repo = self._parse_repo_url(repo_url)
        headers = self._get_headers(token)

        async with httpx.AsyncClient() as client:
            # GitHub code search query format
            query = f"{pattern} repo:{owner}/{repo}"

            response = await client.get(
                f"{self.base_url}/search/code",
                headers=headers,
                params={"q": query, "per_page": max_results},
                timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                results = []

                for item in data.get("items", []):
                    results.append({
                        "file": item.get("name"),
                        "path": item.get("path"),
                        "url": item.get("html_url"),
                        "repository": item.get("repository", {}).get("full_name")
                    })

                return {
                    "total_count": data.get("total_count", 0),
                    "results": results,
                    "pattern": pattern
                }
            else:
                return {
                    "error": f"GitHub API error: {response.status_code}",
                    "message": response.text
                }

    async def get_file_content(
        self,
        repo_url: str,
        file_path: str,
        token: str
    ) -> Optional[str]:
        """
        Get raw content of a file from the repository.

        Args:
            repo_url: GitHub repository URL
            file_path: Path to the file in the repository
            token: GitHub Personal Access Token
        """
        owner, repo = self._parse_repo_url(repo_url)
        headers = self._get_headers(token)

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo}/contents/{file_path}",
                headers=headers,
                timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                # Content is base64 encoded
                content = data.get("content", "")
                if content:
                    return base64.b64decode(content).decode("utf-8")
            return None

    async def get_repo_info(self, repo_url: str, token: str) -> Dict[str, Any]:
        """
        Get repository information.

        Args:
            repo_url: GitHub repository URL
            token: GitHub Personal Access Token
        """
        owner, repo = self._parse_repo_url(repo_url)
        headers = self._get_headers(token)

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo}",
                headers=headers,
                timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "name": data.get("name"),
                    "full_name": data.get("full_name"),
                    "description": data.get("description"),
                    "language": data.get("language"),
                    "default_branch": data.get("default_branch"),
                    "url": data.get("html_url")
                }
            else:
                return {"error": f"Failed to get repo info: {response.status_code}"}

    async def get_file_tree(
        self,
        repo_url: str,
        token: str,
        path: str = ""
    ) -> List[Dict]:
        """
        Get file tree of a repository or subdirectory.

        Args:
            repo_url: GitHub repository URL
            token: GitHub Personal Access Token
            path: Path within the repository (default: root)
        """
        owner, repo = self._parse_repo_url(repo_url)
        headers = self._get_headers(token)

        async with httpx.AsyncClient() as client:
            url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}" if path else f"{self.base_url}/repos/{owner}/{repo}/contents"

            response = await client.get(
                url,
                headers=headers,
                timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return [
                        {
                            "name": item.get("name"),
                            "path": item.get("path"),
                            "type": item.get("type"),  # "file" or "dir"
                            "size": item.get("size", 0)
                        }
                        for item in data
                    ]
            return []

    async def search_and_get_context(
        self,
        repo_url: str,
        error_patterns: List[str],
        token: str
    ) -> str:
        """
        Search for error patterns and return code context.
        Combines search and file content retrieval.

        Args:
            repo_url: GitHub repository URL
            error_patterns: List of patterns to search for
            token: GitHub Personal Access Token
        """
        context_parts = []

        for pattern in error_patterns[:3]:  # Limit to top 3 patterns
            search_result = await self.search_code(repo_url, pattern, token, max_results=3)

            if "results" in search_result:
                for result in search_result["results"][:2]:  # Top 2 files per pattern
                    file_path = result.get("path")
                    if file_path:
                        content = await self.get_file_content(repo_url, file_path, token)
                        if content:
                            # Truncate content for context
                            truncated = content[:1500] if len(content) > 1500 else content
                            context_parts.append(f"// File: {file_path}\n{truncated}")

        return "\n\n".join(context_parts) if context_parts else ""
