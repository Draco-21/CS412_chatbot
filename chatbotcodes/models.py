from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class QueryAnalysis:
    code_language: Optional[str]
    frameworks: List[str]
    topic_category: str
    query_type: str  # 'code', 'explanation', 'both'
    keywords: List[str]
    year_level: str
    algorithm_type: Optional[str] = None  # Added field for specific algorithm type
    algorithm_complexity: Optional[Dict[str, str]] = None  # Added field for complexity info

@dataclass
class ResourceMatch:
    title: str
    purpose: str
    context: Optional[str]
    code: str
    language: str
    topics: List[str]
    score: float
    url: Optional[str] = None

@dataclass
class GeneratedResponse:
    content: str
    source_type: str  # 'local' or 'api'
    resources_used: List[str]
    confidence_score: float 