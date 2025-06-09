import re
from typing import Dict, List, Optional, Tuple
from .models import QueryAnalysis

class QueryAnalyzer:
    def __init__(self):
        self.language_patterns = {
            'python': r'python|django|flask|pandas|numpy|tensorflow|pytorch',
            'java': r'java|spring|springboot|hibernate|jdbc',
            'cpp': r'c\+\+|cpp|c plus plus',
            'javascript': r'javascript|js|node|react|angular|vue',
            'sql': r'sql|mysql|postgresql|database query'
        }
        
        self.algorithm_patterns = {
            'sorting': {
                'bubble_sort': r'bubble\s*sort',
                'merge_sort': r'merge\s*sort',
                'quick_sort': r'quick\s*sort',
                'insertion_sort': r'insertion\s*sort',
                'selection_sort': r'selection\s*sort',
                'heap_sort': r'heap\s*sort'
            },
            'searching': {
                'binary_search': r'binary\s*search',
                'linear_search': r'linear\s*search',
                'depth_first_search': r'depth\s*first|dfs',
                'breadth_first_search': r'breadth\s*first|bfs'
            },
            'graph': {
                'dijkstra': r'dijkstra',
                'kruskal': r'kruskal',
                'prim': r'prim\'?s',
                'floyd_warshall': r'floyd\s*warshall'
            }
        }
        
        self.algorithm_complexities = {
            'merge_sort': {
                'time': 'O(n log n)',
                'space': 'O(n)',
                'description': 'Divide and conquer algorithm that recursively divides array into two halves, sorts them and merges them'
            },
            'quick_sort': {
                'time': 'O(n log n) average, O(n²) worst',
                'space': 'O(log n)',
                'description': 'Divide and conquer algorithm that picks a pivot and partitions array around it'
            },
            'bubble_sort': {
                'time': 'O(n²)',
                'space': 'O(1)',
                'description': 'Simple comparison sort that repeatedly steps through the list, compares adjacent elements and swaps them if needed'
            }
        }
        
        self.framework_patterns = {
            'web': [
                'django', 'flask', 'spring', 'asp.net', 'react', 'angular',
                'vue', 'express', 'fastapi'
            ],
            'ml': [
                'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'pandas',
                'numpy', 'machine learning', 'deep learning'
            ],
            'database': [
                'hibernate', 'jdbc', 'sqlalchemy', 'mongoose', 'prisma',
                'mysql', 'postgresql', 'mongodb'
            ],
            'testing': [
                'junit', 'pytest', 'jest', 'selenium', 'cypress'
            ]
        }
        
        self.category_patterns = {
            'algorithm': r'algorithm|sort|search|graph|tree|complexity',
            'data_structure': r'array|list|stack|queue|tree|graph|hash|heap',
            'web_dev': r'web|html|css|api|rest|http|frontend|backend',
            'database': r'database|sql|query|table|join|crud',
            'machine_learning': r'machine learning|neural|deep learning|ai|train|model',
            'networking': r'network|socket|protocol|tcp|udp|http|api',
            'security': r'security|auth|encryption|token|jwt|oauth'
        }
        
        self.query_type_patterns = {
            'code': r'code|implement|write|create|program|function|class|method',
            'explanation': r'explain|describe|what|how|why|concept|difference|compare',
            'error': r'error|bug|fix|issue|problem|debug|not working'
        }

    def detect_algorithm(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Detect specific algorithm type from query."""
        query_lower = query.lower()
        
        for category, algorithms in self.algorithm_patterns.items():
            for algo_name, pattern in algorithms.items():
                if re.search(pattern, query_lower):
                    return category, algo_name
        return None, None

    def analyze_query(self, query: str, year_level: str) -> QueryAnalysis:
        """Analyze the user's query to understand what they're asking for."""
        query_lower = query.lower()
        
        # Determine language based on year level first
        year_level_languages = {
            "Year 1 Certificate": "c++",
            "Year 2 Diploma": "java", 
            "Year 3 Degree": "python",
            "Year 4 Postgraduate Diploma": "python"
        }
        code_language = year_level_languages.get(year_level)
        
        # Only override if explicitly specified in query
        for lang, pattern in self.language_patterns.items():
            if re.search(pattern, query_lower):
                code_language = lang
                break
        
        # Detect algorithm type
        algo_category, algo_type = self.detect_algorithm(query_lower)
        
        # Get algorithm complexity if available
        complexity_info = self.algorithm_complexities.get(algo_type) if algo_type else None
        
        # Detect frameworks
        frameworks = []
        for category, framework_list in self.framework_patterns.items():
            for framework in framework_list:
                if framework in query_lower:
                    frameworks.append(framework)
        
        # Detect topic category
        topic_category = 'general'
        if algo_category:
            topic_category = 'algorithm'
        else:
            for category, pattern in self.category_patterns.items():
                if re.search(pattern, query_lower):
                    topic_category = category
                    break
        
        # Determine query type
        has_code_pattern = bool(re.search(self.query_type_patterns['code'], query_lower))
        has_explanation_pattern = bool(re.search(self.query_type_patterns['explanation'], query_lower))
        
        if has_code_pattern and has_explanation_pattern:
            query_type = 'both'
        elif has_code_pattern:
            query_type = 'code'
        else:
            query_type = 'explanation'
        
        # Extract keywords (excluding common words)
        common_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in re.findall(r'\w+', query_lower) 
                   if word not in common_words and len(word) > 2]
        
        return QueryAnalysis(
            code_language=code_language,
            frameworks=frameworks,
            topic_category=topic_category,
            query_type=query_type,
            keywords=keywords,
            year_level=year_level,
            algorithm_type=algo_type,
            algorithm_complexity=complexity_info
        )

    def get_search_parameters(self, analysis: QueryAnalysis) -> Dict:
        """Convert query analysis into search parameters for the resource retriever."""
        return {
            'language': analysis.code_language,
            'frameworks': analysis.frameworks,
            'category': analysis.topic_category,
            'query_type': analysis.query_type,
            'keywords': analysis.keywords,
            'year_level': analysis.year_level,
            'algorithm_type': analysis.algorithm_type,
            'algorithm_complexity': analysis.algorithm_complexity
        } 