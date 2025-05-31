"""Enhanced capability discovery engine for finding relevant open source tools and libraries."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .config import Config


@dataclass
class DiscoveredCapability:
    """A discovered capability with metadata."""
    name: str
    description: str
    source: str  # github, pypi, manual, etc.
    url: Optional[str]
    language: str
    stars: int
    complexity_score: float
    relevance_score: float
    keywords: List[str]
    category: str
    last_updated: Optional[str]


class GitHubCapabilityDiscoverer:
    """Discovers capabilities by searching GitHub repositories."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.github_token = config.github_token
        self.base_url = "https://api.github.com"
        
        # Repository categories for different domains
        self.domain_keywords = {
            "data_analysis": ["pandas", "numpy", "data-analysis", "csv", "excel", "statistics"],
            "web_apis": ["httpx", "requests", "fastapi", "flask", "api", "rest", "graphql"],
            "file_processing": ["pathlib", "pdf", "docx", "text-processing", "file-utils"],
            "mathematical": ["numpy", "scipy", "sympy", "math", "calculation", "algorithm"],
            "visualization": ["matplotlib", "plotly", "seaborn", "charts", "graphs"],
            "financial": ["yfinance", "alpha-vantage", "trading", "stocks", "finance", "quantlib"],
            "machine_learning": ["scikit-learn", "tensorflow", "pytorch", "ml", "ai"],
            "database": ["sqlalchemy", "sqlite", "postgresql", "mongodb", "database"]
        }
    
    async def discover_capabilities(
        self, 
        task_description: str, 
        max_results: int = 10
    ) -> List[DiscoveredCapability]:
        """Discover relevant capabilities for the given task."""
        
        if not HAS_HTTPX:
            self.logger.warning("httpx not available, using fallback discovery")
            return await self._fallback_discovery(task_description)
        
        try:
            # Extract keywords and determine domain
            keywords = self._extract_keywords(task_description)
            domain = self._classify_domain(keywords)
            
            # Search for relevant repositories
            repositories = await self._search_github_repositories(keywords, domain, max_results)
            
            # Convert to DiscoveredCapability objects
            capabilities = []
            for repo in repositories:
                capability = await self._analyze_repository(repo, keywords, domain)
                if capability:
                    capabilities.append(capability)
            
            # Sort by relevance score
            capabilities.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return capabilities[:max_results]
            
        except Exception as e:
            self.logger.error(f"Error discovering capabilities: {str(e)}")
            return await self._fallback_discovery(task_description)
    
    def _extract_keywords(self, task_description: str) -> List[str]:
        """Extract relevant keywords from task description."""
        # Convert to lowercase and split
        words = re.findall(r'\b\w+\b', task_description.lower())
        
        # Filter out common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'i', 'need', 'want', 'can', 'should', 'would', 'could', 'have', 'has', 'is',
            'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does', 'did', 'will', 'would'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add domain-specific technical terms
        technical_terms = {
            'api', 'http', 'rest', 'json', 'csv', 'pdf', 'sql', 'database', 'ml',
            'ai', 'analysis', 'visualization', 'chart', 'graph', 'data', 'file',
            'math', 'calculation', 'algorithm', 'trading', 'stock', 'finance'
        }
        
        for word in words:
            if word in technical_terms:
                keywords.append(word)
        
        return list(set(keywords))
    
    def _classify_domain(self, keywords: List[str]) -> str:
        """Classify the task domain based on keywords."""
        domain_scores = {}
        
        for domain, domain_keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                for domain_keyword in domain_keywords:
                    if keyword in domain_keyword or domain_keyword in keyword:
                        score += 1
            domain_scores[domain] = score
        
        # Return domain with highest score, or 'general' if no clear match
        if not domain_scores or max(domain_scores.values()) == 0:
            return 'general'
        
        return max(domain_scores, key=domain_scores.get)
    
    async def _search_github_repositories(
        self, 
        keywords: List[str], 
        domain: str, 
        max_results: int
    ) -> List[Dict]:
        """Search GitHub for relevant repositories."""
        
        # Build search query
        search_terms = keywords[:3]  # Use top 3 keywords
        if domain != 'general':
            domain_keywords = self.domain_keywords.get(domain, [])
            search_terms.extend(domain_keywords[:2])
        
        query = ' '.join(search_terms)
        
        # Add filters for quality repositories
        filters = [
            "language:python",
            "stars:>10",
            "size:>100"  # KB
        ]
        
        full_query = f"{query} {' '.join(filters)}"
        
        headers = {}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/search/repositories",
                    params={
                        "q": full_query,
                        "sort": "stars",
                        "order": "desc",
                        "per_page": min(max_results * 2, 30)  # Get more than needed for filtering
                    },
                    headers=headers,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("items", [])
                else:
                    self.logger.warning(f"GitHub API returned status {response.status_code}")
                    return []
                    
            except Exception as e:
                self.logger.error(f"Error searching GitHub: {str(e)}")
                return []
    
    async def _analyze_repository(
        self, 
        repo: Dict, 
        keywords: List[str], 
        domain: str
    ) -> Optional[DiscoveredCapability]:
        """Analyze a repository and create a DiscoveredCapability."""
        
        try:
            name = repo.get("name", "")
            description = repo.get("description", "")
            stars = repo.get("stargazers_count", 0)
            language = repo.get("language", "Unknown")
            url = repo.get("html_url", "")
            updated_at = repo.get("updated_at", "")
            
            # Calculate complexity score based on repository characteristics
            complexity_score = self._calculate_complexity_score(repo)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(
                name, description, keywords, domain, stars
            )
            
            # Extract keywords from repository
            repo_keywords = self._extract_repo_keywords(name, description, repo.get("topics", []))
            
            return DiscoveredCapability(
                name=name,
                description=description or f"GitHub repository: {name}",
                source="github",
                url=url,
                language=language,
                stars=stars,
                complexity_score=complexity_score,
                relevance_score=relevance_score,
                keywords=repo_keywords,
                category=domain,
                last_updated=updated_at
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing repository: {str(e)}")
            return None
    
    def _calculate_complexity_score(self, repo: Dict) -> float:
        """Calculate complexity score based on repository metrics."""
        size = repo.get("size", 0)  # KB
        forks = repo.get("forks_count", 0)
        open_issues = repo.get("open_issues_count", 0)
        
        # Normalize scores (0-1 scale)
        size_score = min(size / 10000, 1.0)  # Large repos are more complex
        forks_score = min(forks / 100, 1.0)  # More forks suggest complexity
        issues_score = min(open_issues / 50, 1.0)  # More issues suggest complexity
        
        # Weighted average
        complexity = (size_score * 0.4 + forks_score * 0.3 + issues_score * 0.3)
        
        return round(complexity, 2)
    
    def _calculate_relevance_score(
        self, 
        name: str, 
        description: str, 
        keywords: List[str], 
        domain: str, 
        stars: int
    ) -> float:
        """Calculate relevance score for the repository."""
        
        text = f"{name} {description}".lower()
        
        # Keyword matching score
        keyword_matches = sum(1 for keyword in keywords if keyword in text)
        keyword_score = min(keyword_matches / len(keywords), 1.0) if keywords else 0.0
        
        # Domain relevance score
        domain_keywords = self.domain_keywords.get(domain, [])
        domain_matches = sum(1 for keyword in domain_keywords if keyword in text)
        domain_score = min(domain_matches / len(domain_keywords), 1.0) if domain_keywords else 0.0
        
        # Popularity score (stars)
        popularity_score = min(stars / 1000, 1.0)  # Cap at 1000 stars
        
        # Weighted average
        relevance = (keyword_score * 0.5 + domain_score * 0.3 + popularity_score * 0.2)
        
        return round(relevance, 2)
    
    def _extract_repo_keywords(self, name: str, description: str, topics: List[str]) -> List[str]:
        """Extract keywords from repository metadata."""
        keywords = set()
        
        # Add name parts
        name_parts = re.findall(r'\b\w+\b', name.lower())
        keywords.update(name_parts)
        
        # Add description words
        if description:
            desc_words = re.findall(r'\b\w+\b', description.lower())
            keywords.update(desc_words)
        
        # Add topics
        keywords.update(topic.lower() for topic in topics)
        
        # Filter out generic words
        filtered_keywords = [
            keyword for keyword in keywords 
            if len(keyword) > 2 and keyword not in {'the', 'and', 'for', 'with', 'python'}
        ]
        
        return filtered_keywords[:10]  # Limit to top 10
    
    async def _fallback_discovery(self, task_description: str) -> List[DiscoveredCapability]:
        """Fallback discovery when GitHub API is not available."""
        
        # Common Python libraries for different domains
        fallback_capabilities = {
            "data_analysis": [
                DiscoveredCapability(
                    name="pandas",
                    description="Powerful data structures and data analysis tools",
                    source="fallback",
                    url="https://pandas.pydata.org/",
                    language="python",
                    stars=0,
                    complexity_score=0.7,
                    relevance_score=0.9,
                    keywords=["data", "analysis", "csv", "excel"],
                    category="data_analysis",
                    last_updated=None
                ),
                DiscoveredCapability(
                    name="numpy",
                    description="Fundamental package for scientific computing",
                    source="fallback",
                    url="https://numpy.org/",
                    language="python",
                    stars=0,
                    complexity_score=0.6,
                    relevance_score=0.8,
                    keywords=["numerical", "array", "math"],
                    category="data_analysis",
                    last_updated=None
                )
            ],
            "web_apis": [
                DiscoveredCapability(
                    name="httpx",
                    description="A fully featured HTTP client for Python 3",
                    source="fallback",
                    url="https://www.python-httpx.org/",
                    language="python",
                    stars=0,
                    complexity_score=0.5,
                    relevance_score=0.9,
                    keywords=["http", "api", "requests"],
                    category="web_apis",
                    last_updated=None
                )
            ],
            "financial": [
                DiscoveredCapability(
                    name="yfinance",
                    description="Download market data from Yahoo! Finance API",
                    source="fallback",
                    url="https://github.com/ranaroussi/yfinance",
                    language="python",
                    stars=0,
                    complexity_score=0.4,
                    relevance_score=0.9,
                    keywords=["finance", "stocks", "trading", "yahoo"],
                    category="financial",
                    last_updated=None
                )
            ]
        }
        
        # Determine domain and return relevant capabilities
        keywords = self._extract_keywords(task_description)
        domain = self._classify_domain(keywords)
        
        return fallback_capabilities.get(domain, [])


class CapabilityMatchingEngine:
    """Engine for matching discovered capabilities to task requirements."""
    
    def __init__(self, config: Config):
        self.config = config
        self.discoverer = GitHubCapabilityDiscoverer(config)
        self.logger = logging.getLogger(__name__)
    
    async def find_best_capabilities(
        self, 
        task_description: str, 
        max_results: int = 5
    ) -> List[DiscoveredCapability]:
        """Find the best capabilities for a given task."""
        
        # Discover capabilities
        capabilities = await self.discoverer.discover_capabilities(task_description, max_results * 2)
        
        if not capabilities:
            return []
        
        # Apply additional filtering and ranking
        filtered_capabilities = self._filter_capabilities(capabilities, task_description)
        
        # Return top results
        return filtered_capabilities[:max_results]
    
    def _filter_capabilities(
        self, 
        capabilities: List[DiscoveredCapability], 
        task_description: str
    ) -> List[DiscoveredCapability]:
        """Apply additional filtering to discovered capabilities."""
        
        filtered = []
        
        for capability in capabilities:
            # Skip very low relevance
            if capability.relevance_score < 0.1:
                continue
            
            # Skip deprecated or unmaintained (no updates in 2+ years for high complexity)
            if capability.complexity_score > 0.7 and capability.last_updated:
                # Would need date parsing for real implementation
                pass
            
            # Prefer Python libraries for easier integration
            if capability.language.lower() == "python":
                capability.relevance_score += 0.1
            
            filtered.append(capability)
        
        return filtered
    
    def generate_integration_suggestions(
        self, 
        capabilities: List[DiscoveredCapability]
    ) -> List[str]:
        """Generate suggestions for integrating discovered capabilities."""
        
        suggestions = []
        
        for capability in capabilities:
            if capability.source == "github":
                suggestions.append(
                    f"Consider integrating {capability.name}: {capability.description} "
                    f"(Stars: {capability.stars}, Relevance: {capability.relevance_score})"
                )
            else:
                suggestions.append(
                    f"Standard library option: {capability.name} - {capability.description}"
                )
        
        return suggestions