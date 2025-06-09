import unittest
import sys
import os
from pathlib import Path
import numpy as np

# Add parent directory to path to import from retrievalAlgorithms
sys.path.append(str(Path(__file__).parent.parent))
from retrievalAlgorithms.resource_retriever import SmartResourceRetriever

class TestSemanticSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test resources and initialize SmartResourceRetriever."""
        cls.retriever = SmartResourceRetriever()
        
        # Create test content
        cls.test_content = {
            'python_basics': {
                'title': 'Python Basic Loop Examples',
                'language': 'python',
                'purpose': 'Demonstrate basic loop structures in Python',
                'code': '''
def example_loops():
    # For loop example
    for i in range(5):
        print(i)
    
    # While loop example
    count = 0
    while count < 5:
        print(count)
        count += 1
''',
                'context': 'Basic programming concepts for beginners',
                'topics': ['fundamentals', 'control_flow'],
                'text': 'Python loop examples including for loops and while loops with basic counter examples.'
            },
            'cpp_algorithms': {
                'title': 'C++ Sorting Algorithms',
                'language': 'cpp',
                'purpose': 'Implementation of common sorting algorithms',
                'code': '''
void bubbleSort(int arr[], int n) {
    for(int i = 0; i < n-1; i++)
        for(int j = 0; j < n-i-1; j++)
            if(arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
}
''',
                'context': 'Algorithm implementation examples',
                'topics': ['algorithms', 'sorting'],
                'text': 'Implementation of bubble sort algorithm in C++ with nested loops and swap operations.'
            },
            'web_dev': {
                'title': 'Basic Web Development',
                'language': 'javascript',
                'purpose': 'Introduction to web development',
                'code': '''
function handleClick() {
    document.getElementById("demo").innerHTML = "Hello World!";
}
''',
                'context': 'Web development fundamentals',
                'topics': ['web_dev', 'frontend'],
                'text': 'Basic JavaScript DOM manipulation and event handling examples.'
            }
        }

    def test_text_preprocessing(self):
        """Test the enhanced text preprocessing functionality."""
        test_text = "Python programming with loops and functions"
        tokens = self.retriever._preprocess_text(test_text)
        
        # Check if basic tokens are present
        self.assertTrue('python' in tokens)
        self.assertTrue('program' in tokens)
        self.assertTrue('loop' in tokens)
        self.assertTrue('function' in tokens)
        
        # Check if bigrams are present
        self.assertTrue('python_program' in tokens or 'program_loop' in tokens)

    def test_semantic_similarity(self):
        """Test semantic similarity calculation."""
        text1 = "Python loop examples"
        text2 = "Examples of iteration in Python"
        text3 = "Database management systems"
        
        # Similar texts should have higher similarity
        sim1 = self.retriever._calculate_semantic_similarity(text1, text2)
        sim2 = self.retriever._calculate_semantic_similarity(text1, text3)
        
        self.assertGreater(sim1, sim2)  # Similar texts should have higher score
        self.assertGreaterEqual(sim1, 0.0)  # Scores should be normalized
        self.assertLessEqual(sim1, 1.0)

    def test_topic_detection(self):
        """Test hierarchical topic detection."""
        query = "How to implement sorting algorithms in C++"
        detected = self.retriever._detect_topics_and_language(query)
        
        # Check if correct topics are detected
        self.assertTrue('algorithms' in detected['topics'])
        self.assertTrue('cpp' in detected['languages'])
        
        # Check hierarchical detection
        algo_matches = detected['topics']['algorithms']
        self.assertTrue(any(match['subcategory'] == 'sorting' for match in algo_matches))

    def test_relevance_scoring(self):
        """Test the enhanced relevance scoring system."""
        query = "Python loop examples for beginners"
        
        # Test with Python basics content
        score1 = self.retriever._calculate_relevance_score(
            self.test_content['python_basics'],
            query,
            self.retriever._detect_topics_and_language(query)
        )
        
        # Test with C++ algorithms content
        score2 = self.retriever._calculate_relevance_score(
            self.test_content['cpp_algorithms'],
            query,
            self.retriever._detect_topics_and_language(query)
        )
        
        # Python basics should be more relevant for this query
        self.assertGreater(score1, score2)
        
        # Scores should be normalized
        self.assertGreaterEqual(score1, 0.0)
        self.assertLessEqual(score1, 1.0)
        self.assertGreaterEqual(score2, 0.0)
        self.assertLessEqual(score2, 1.0)

    def test_language_specific_search(self):
        """Test language-specific search capabilities."""
        queries = [
            ("javascript DOM manipulation", "javascript"),
            ("C++ vector implementation", "cpp"),
            ("Python list comprehension", "python")
        ]
        
        for query, expected_lang in queries:
            detected = self.retriever._detect_topics_and_language(query)
            self.assertTrue(expected_lang in detected['languages'])

    def test_beginner_content_detection(self):
        """Test detection and scoring of beginner-friendly content."""
        query = "simple programming examples for beginners"
        detected = self.retriever._detect_topics_and_language(query)
        
        # Should detect it's a beginner request
        self.assertTrue(
            any('basic' in match['subcategory'] 
                for matches in detected['topics'].values() 
                for match in matches)
        )

    def test_code_quality_scoring(self):
        """Test code quality assessment in scoring."""
        content_with_comments = {
            'title': 'Well-documented Code',
            'language': 'python',
            'purpose': 'Example with good documentation',
            'code': '''
# This function demonstrates good documentation
def well_documented():
    """
    This is a docstring explaining the function
    """
    # Initialize counter
    count = 0
    # Increment and return
    return count + 1
''',
            'context': 'Code quality example',
            'topics': ['fundamentals'],
            'text': 'Example of well-documented code with comments and docstring.'
        }
        
        content_without_comments = {
            'title': 'Poorly documented Code',
            'language': 'python',
            'purpose': 'Example without documentation',
            'code': '''
def poorly_documented():
    count = 0
    return count + 1
''',
            'context': 'Code quality example',
            'topics': ['fundamentals'],
            'text': 'Example of code without proper documentation.'
        }
        
        query = "well documented python code examples"
        detected = self.retriever._detect_topics_and_language(query)
        
        score1 = self.retriever._calculate_relevance_score(
            content_with_comments, query, detected)
        score2 = self.retriever._calculate_relevance_score(
            content_without_comments, query, detected)
        
        self.assertGreater(score1, score2)

    def test_multi_topic_detection(self):
        """Test detection of multiple topics in a single query."""
        query = "implementing sorting algorithms with data structures in C++"
        detected = self.retriever._detect_topics_and_language(query)
        
        # Should detect both algorithms and data structures
        self.assertTrue('algorithms' in detected['topics'])
        self.assertTrue('data_structures' in detected['topics'])
        self.assertTrue('cpp' in detected['languages'])

    def test_semantic_search_context(self):
        """Test if search results maintain proper context."""
        query = "web development with JavaScript events"
        results = self.retriever.search_resources(query, "Year 1 Certificate")
        
        if results:
            top_result = results[0]
            # Check if context is maintained
            self.assertTrue(
                any(topic.startswith('web') for topic in top_result.topics) or
                'javascript' in top_result.language.lower()
            )

    def test_advanced_topic_search(self):
        """Test search functionality for advanced topics."""
        advanced_queries = [
            "advanced algorithm optimization techniques",
            "complex data structure implementations",
            "advanced web development patterns"
        ]
        
        for query in advanced_queries:
            detected = self.retriever._detect_topics_and_language(query)
            # Should detect it's an advanced topic
            self.assertTrue(
                any('advanced' in str(matches) or 'complex' in str(matches)
                    for matches in detected['topics'].values())
            )

if __name__ == '__main__':
    unittest.main() 