import os
import re
import pdfplumber  # For reading PDF files
import docx        # For reading DOCX files
import pptx        # For reading PPTX files
import PyPDF2
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from functools import lru_cache
import concurrent.futures
import hashlib
from threading import Lock
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

RESOURCES_BASE_FOLDER = "cleaned_resources"
YEAR_FOLDER_MAP = {
    "Year 1 Certificate": "Y1",
    "Year 2 Diploma": "Y2",
    "Year 3 Degree": "Y3",
    "Year 4 Postgraduate Diploma": "Y4"
}

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('resource_retrieval.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# --- Text Extraction Functions (No changes needed here) ---

def extract_text_from_pdf(file_path: str) -> str | None:
    """Extracts text content from a PDF file."""
    try:
        with pdfplumber.open(file_path) as pdf:
            text_content = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
            return text_content.lower()
    except Exception as e:
        print(f"ResourceRetriever: Error reading PDF file {file_path}: {e}")
        return None

def extract_text_from_txt_or_code(file_path: str) -> str | None:
    """Extracts text content from a .txt, .md, .cpp, .h, .js, .py file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().lower()
    except Exception as e:
        print(f"ResourceRetriever: Error reading text/code file {file_path}: {e}")
        return None

def extract_text_from_docx(file_path: str) -> str | None:
    """Extracts text content from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text).lower()
    except Exception as e:
        print(f"ResourceRetriever: Error reading DOCX file {file_path}: {e}")
        return None

def extract_text_from_pptx(file_path: str) -> str | None:
    """Extracts text content from a PPTX file."""
    try:
        prs = pptx.Presentation(file_path)
        full_text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    full_text.append(shape.text)
        return '\n'.join(full_text).lower()
    except Exception as e:
        print(f"ResourceRetriever: Error reading PPTX file {file_path}: {e}")
        return None

# --- NEW Helper Function to Search a Single Folder ---

def search_single_folder(folder_path: str, keywords: list[str]) -> str | None:
    """
    Searches recursively within a single folder path for files matching keywords.
    Returns the full content of the first matching file found.
    """
    if not os.path.isdir(folder_path):
        print(f"ResourceRetriever: Search path {folder_path} not found or not a directory.")
        return None

    try:
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                content = None
                file_ext = filename.lower().split('.')[-1] if '.' in filename else ''

                # Extract text based on file type
                if file_ext == "pdf":
                    content = extract_text_from_pdf(file_path)
                elif file_ext in ["txt", "md", "cpp", "h", "js", "py", "gitignore", "log"]:
                    content = extract_text_from_txt_or_code(file_path)
                elif file_ext == "docx":
                    content = extract_text_from_docx(file_path)
                elif file_ext in ["ppt", "pptx"]:
                    content = extract_text_from_pptx(file_path)

                if content:
                    # Check if ANY keyword is present
                    found_keywords_count = sum(1 for keyword in keywords if keyword in content)
                    if found_keywords_count > 0:
                        print(f"ResourceRetriever: Found {found_keywords_count} keyword(s) in '{file_path}'.")
                        return content # Return full content of the first match

    except Exception as e:
        print(f"ResourceRetriever: Error during search in {folder_path}: {e}")
        return None

    return None # No match found in this folder

# --- UPDATED Main Search Function ---

def find_answer_in_resources(user_message: str, year_level: str) -> str | None:
    """
    Searches for an answer within local resource files, checking the primary
    year level first, then expanding to other year levels.
    """
    print(f"ResourceRetriever: Main search for '{user_message}', primary year: {year_level}.")

    keywords = [word for word in re.split(r'\W+', user_message.lower()) if word]
    if not keywords:
        print("ResourceRetriever: No valid keywords in user message.")
        return None
    
    print(f"ResourceRetriever: Using keywords: {keywords}")

    # --- Determine Search Order ---
    primary_folder_name = YEAR_FOLDER_MAP.get(year_level)
    all_folder_names = list(YEAR_FOLDER_MAP.values())
    
    search_order_names = []
    
    # 1. Add primary folder first if it exists
    if primary_folder_name:
        search_order_names.append(primary_folder_name)
    else:
        print(f"ResourceRetriever: Warning - No folder mapping for {year_level}")

    # 2. Add all other folders (excluding the primary one if it was added)
    for name in all_folder_names:
        if name not in search_order_names:
            search_order_names.append(name)

    print(f"ResourceRetriever: Search order: {search_order_names}")

    # --- Execute Search in Order ---
    for folder_name in search_order_names:
        folder_path = os.path.join(RESOURCES_BASE_FOLDER, folder_name)
        print(f"ResourceRetriever: Now searching in: {folder_path}")
        
        result = search_single_folder(folder_path, keywords)
        
        if result:
            print(f"ResourceRetriever: Match found in {folder_path}.")
            return result # Return the first match we find

    # If we get here, no match was found in any folder
    print(f"ResourceRetriever: No match found in any folder for '{user_message}'.")
    return None

# --- Example Usage (remains the same for testing) ---
if __name__ == "__main__":
    # ... (Keep your example usage/dummy file creation if needed) ...
    print("Testing with 'C++ loops'")
    result = find_answer_in_resources("C++ loops", "Year 1 Certificate")
    if result:
        print("\n--- Search Result Found (Showing first 500 chars) ---")
        print(result[:500] + "...")
    else:
        print("\nNo local result found.")

@dataclass
class ResourceMatch:
    title: str
    language: str
    purpose: str
    code: str
    context: str
    source_file: str
    topics: List[str]
    score: float = 0.0

class SmartResourceRetriever:
    def __init__(self, resources_dir=None):
        logger.info("Initializing SmartResourceRetriever")
        self.resources_dir = resources_dir or RESOURCES_BASE_FOLDER
        self._content_cache = {}  # File path -> content cache
        self._cache_lock = Lock()  # Thread-safe cache access
        self._keyword_cache = {}   # Query -> keywords cache
        
        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Topic categories for search optimization
        self.topic_categories = {
            'fundamentals': ['basic', 'syntax', 'variable', 'loop', 'condition'],
            'data_structures': ['array', 'list', 'tree', 'graph', 'stack', 'queue'],
            'algorithms': ['sort', 'search', 'recursion', 'complexity'],
            'web_dev': ['html', 'css', 'javascript', 'api'],
            'database': ['sql', 'query', 'table', 'database'],
            'networking': ['tcp', 'ip', 'socket', 'protocol'],
            'machine_learning': ['ml', 'neural', 'training', 'model']
        }
        
        # Programming languages for search optimization
        self.programming_languages = {
            'python': ['python', 'django', 'flask', 'pandas'],
            'cpp': ['c++', 'cpp', 'vector', 'iostream'],
            'java': ['java', 'spring', 'hibernate'],
            'javascript': ['js', 'node', 'react', 'angular']
        }
        
        # Load and cache all resources on initialization
        self._load_all_resources()
        
        logger.info(f"SmartResourceRetriever initialized with {len(self._content_cache)} resources")

    def _load_all_resources(self):
        """Load all JSON resources into memory cache."""
        try:
            # Get absolute path to resources directory
            resources_root = Path(os.path.abspath(self.resources_dir))
            if not resources_root.exists():
                logger.error(f"Resources directory not found: {resources_root}")
                return

            for json_file in resources_root.rglob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        self._content_cache[str(json_file)] = content
                        logger.debug(f"Loaded resource: {json_file}")
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {str(e)}")

        except Exception as e:
            logger.error(f"Error loading resources: {str(e)}")

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text into lemmatized tokens."""
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words and token.isalnum()]
        
        return tokens

    def _detect_topics_and_language(self, query: str) -> Dict[str, List[str]]:
        """Detect programming languages and topics from the query."""
        query_lower = query.lower()
        detected = {
            'languages': [],
            'topics': []
        }
        
        # Enhanced language detection
        for lang, keywords in self.programming_languages.items():
            # Check for explicit language mentions
            if any(keyword in query_lower for keyword in keywords):
                detected['languages'].append(lang)
                break
        
        # If no language detected but query contains code-like patterns
        if not detected['languages']:
            if any(pattern in query_lower for pattern in 
                  ['program', 'code', 'function', 'class', 'algorithm']):
                # Default to C++ for basic programming queries
                detected['languages'].append('cpp')
        
        # Enhanced topic detection
        for topic, keywords in self.topic_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                detected['topics'].append(topic)
        
        # Add fundamentals for basic queries
        if 'simple' in query_lower or 'basic' in query_lower or 'example' in query_lower:
            if 'fundamentals' not in detected['topics']:
                detected['topics'].append('fundamentals')
        
        return detected

    def _filter_by_year_level(self, file_path: str, year_level: str) -> bool:
        """Check if a resource matches the year level."""
        year_code = YEAR_FOLDER_MAP.get(year_level)
        if not year_code:
            return True  # If no year specified, include all
        
        # Check if the file is in the correct year folder
        path_parts = Path(file_path).parts
        return any(part == year_code for part in path_parts)

    def _calculate_relevance_score(self, content: Dict, query_tokens: List[str], 
                                 detected_langs: List[str], detected_topics: List[str]) -> float:
        """Calculate how relevant a resource is to the query."""
        score = 0.0
        
        # Special handling for simple program requests
        query_text = ' '.join(query_tokens).lower()
        if 'simple' in query_text and 'program' in query_text:
            # Boost score for basic/simple programs
            if any(word in content['title'].lower() for word in ['simple', 'basic', 'hello', 'first']):
                score += 0.5
            # Penalize complex or advanced topics
            if any(word in content['title'].lower() for word in ['advanced', 'complex', 'expert']):
                score -= 0.3
        
        # Convert content text to tokens for comparison
        content_text = f"{content['title']} {content['purpose']} {content['context']}"
        content_tokens = self._preprocess_text(content_text)
        
        # Score based on token matches in title (higher weight)
        title_tokens = self._preprocess_text(content['title'])
        title_matches = set(query_tokens) & set(title_tokens)
        if title_matches:
            score += 0.4 * (len(title_matches) / len(query_tokens))
        
        # Score based on token matches in purpose and context
        content_matches = set(query_tokens) & set(content_tokens)
        if content_matches:
            score += 0.3 * (len(content_matches) / len(query_tokens))
        
        # Bonus for language match
        if detected_langs and content['language'].lower() in [lang.lower() for lang in detected_langs]:
            score += 0.3
        
        # Bonus for topic match
        if detected_topics and any(topic in content['topics'] for topic in detected_topics):
            score += 0.2
        
        # Bonus for code quality and simplicity
        if content['code'].strip():
            # Basic code quality checks
            has_comments = '//' in content['code'] or '#' in content['code']
            has_structure = '{' in content['code'] or 'def ' in content['code']
            code_lines = len(content['code'].split('\n'))
            
            code_score = 0.1  # Base score for having code
            if has_comments:
                code_score += 0.05
            if has_structure:
                code_score += 0.05
            # For simple program requests, prefer shorter code
            if 'simple' in query_text:
                if 5 <= code_lines <= 15:  # Ideal length for simple examples
                    code_score += 0.1
            else:
                if 5 <= code_lines <= 50:  # Normal ideal length
                    code_score += 0.05
            
            score += code_score
        
        # Bonus for beginner-friendly content
        if any(word in content_text.lower() for word in ['simple', 'basic', 'beginner', 'introduction']):
            score += 0.1
        
        return min(score, 1.0)  # Normalize to 0-1 range

    def search_resources(self, query: str, year_level: str, min_score: float = 0.2) -> List[ResourceMatch]:
        """Search for relevant resources based on the query."""
        logger.info(f"Searching for: '{query}' (year level: {year_level})")
        
        try:
            # Preprocess query
            query_tokens = self._preprocess_text(query)
            detected = self._detect_topics_and_language(query)
            
            logger.debug(f"Query tokens: {query_tokens}")
            logger.debug(f"Detected languages: {detected['languages']}")
            logger.debug(f"Detected topics: {detected['topics']}")
            
            matches = []
            
            # For simple program requests, first try to find exact matches
            if 'simple' in query.lower() and 'program' in query.lower():
                # Look for simple/basic program examples first
                for file_path, content in self._content_cache.items():
                    if (self._filter_by_year_level(file_path, year_level) and
                        any(word in content['title'].lower() for word in ['simple', 'basic', 'hello']) and
                        content['language'].lower() == detected['languages'][0]):
                        
                        match = ResourceMatch(
                            title=content['title'],
                            language=content['language'],
                            purpose=content['purpose'],
                            code=content['code'].strip(),
                            context=content['context'],
                            source_file=content['source_file'],
                            topics=content['topics'],
                            score=0.9  # High score for exact matches
                        )
                        matches.append(match)
                        logger.debug(f"Found simple program match in {file_path}")
                
                if matches:
                    # Sort by code simplicity (fewer lines = simpler)
                    matches.sort(key=lambda x: len(x.code.split('\n')))
                    return matches[:1]  # Return only the simplest match
            
            # If no simple matches found or not a simple program request,
            # proceed with normal search
            for file_path, content in self._content_cache.items():
                try:
                    # Filter by year level
                    if not self._filter_by_year_level(file_path, year_level):
                        continue
                    
                    # Calculate relevance score
                    score = self._calculate_relevance_score(
                        content, 
                        query_tokens,
                        detected['languages'],
                        detected['topics']
                    )
                    
                    if score >= min_score:
                        # Clean and format the content
                        clean_code = content['code'].strip()
                        if not clean_code:
                            continue
                        
                        match = ResourceMatch(
                            title=content['title'],
                            language=content['language'],
                            purpose=content['purpose'],
                            code=clean_code,
                            context=content['context'],
                            source_file=content['source_file'],
                            topics=content['topics'],
                            score=score
                        )
                        matches.append(match)
                        logger.debug(f"Found match in {file_path} with score {score}")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    continue
            
            # Sort by relevance score
            matches.sort(key=lambda x: x.score, reverse=True)
            
            # Limit matches based on query type
            if 'simple' in query.lower():
                matches = matches[:1]  # Only return best match for simple queries
            else:
                matches = matches[:5]  # Return up to 5 matches for other queries
            
            logger.info(f"Found {len(matches)} matches above score {min_score}")
            return matches
            
        except Exception as e:
            logger.error(f"Error during resource search: {str(e)}")
            return []

    def generate_response(self, query: str, matches: List[ResourceMatch], year_level: str) -> Optional[str]:
        """Generate a response based on the search results."""
        if not matches:
            logger.info("No matches found for response generation")
            return None
        
        try:
            # Get the best match
            best_match = matches[0]
            
            # For simple program requests, provide a cleaner, tutorial-style response
            if "simple" in query.lower() and "program" in query.lower():
                response = f"""Let's create a simple {best_match.language.upper()} program that displays "Hello, world!" on the screen. This is the classic first program for almost every programmer.

Step 1: Understanding the Structure

A basic {best_match.language.upper()} program looks like this:

```{best_match.language}
{best_match.code}
```

This program introduces you to:
- Basic {best_match.language.upper()} program structure
- How to output text to the console
- The main() function that every {best_match.language.upper()} program needs"""
            else:
                # For other queries, show a more detailed but still clean response
                response = f"""Here's a {best_match.language.upper()} example for your question:

Purpose: {best_match.purpose}

```{best_match.language}
{best_match.code}
```

Key Points:
- Language: {best_match.language.upper()}
- Topics: {', '.join(best_match.topics)}
{f'- Context: {best_match.context}' if best_match.context else ''}"""

            logger.info("Generated response successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    retriever = SmartResourceRetriever()
    results = retriever.search_resources(
        "How to implement a binary search tree in Python?",
        "Year 2 Diploma"
    )
    if results:
        response = retriever.generate_response(
            "How to implement a binary search tree in Python?",
            results,
            "Year 2 Diploma"
        )
        print(response)
    else:
        print("No relevant resources found.")