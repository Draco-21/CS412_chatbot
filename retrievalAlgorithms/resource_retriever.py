import os
import re
import pdfplumber  # For reading PDF files
import docx        # For reading DOCX files
import pptx        # For reading PPTX files
import PyPDF2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache
import concurrent.futures
import hashlib
from threading import Lock

RESOURCES_BASE_FOLDER = "resources"
YEAR_FOLDER_MAP = {
    "Year 1 Certificate": "Year 1",
    "Year 2 Diploma": "Year 2",
    "Year 3 Degree": "Year 3",
    "Year 4 Postgraduate Diploma": "Year 4"
}

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
    file_path: str
    content: str
    score: float
    context_type: str  # 'framework', 'basic', 'lecture', 'doc', 'practical'
    framework_name: Optional[str] = None

class SmartResourceRetriever:
    def __init__(self):
        self.resources_dir = "resources"
        self._content_cache = {}  # File path -> content cache
        self._cache_lock = Lock()  # Thread-safe cache access
        self._keyword_cache = {}   # Query -> keywords cache
        
        # Define language to framework mappings
        self.framework_mappings = {
            'java': ['springboot', 'spring', 'spring-boot'],
            'python': ['flask', 'django'],
            'web': ['asp.net', 'asp', '.net', 'flask'],
            'network': ['socket', 'cisco']
        }
        
        # Define year level mappings
        self.year_mapping = {
            "Year 1 Certificate": {
                'folder': "Year 1",
                'languages': ['c++'],
                'contexts': {
                    'docs': 'C++ Docs',
                    'lectures': 'CS111 Lectures',
                    'practical': 'Practical'
                }
            },
            "Year 2 Diploma": {
                'folder': "Year 2",
                'languages': ['java', 'asp.net', 'matlab'],
                'frameworks': {
                    'java': ['springboot'],
                    'java_dsa': 'Java DSA',
                    'web': ['asp.net'],
                    'android': ['android studio']
                }
            },
            "Year 3 Degree": {
                'folder': "Year 3",
                'languages': ['python', 'flask'],
                'frameworks': {
                    'python': ['flask'],
                    'network': ['socket', 'cisco']
                }
            },
            "Year 4 Postgraduate Diploma": {
                'folder': "Year 4",
                'languages': ['python', 'ml'],
                'focus': ['machine learning', 'neural networks', 'deep learning']
            }
        }

    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file for cache validation."""
        try:
            return str(os.path.getmtime(file_path))
        except OSError:
            return ""

    def _get_cached_content(self, file_path: str) -> Optional[str]:
        """Get cached content if valid."""
        with self._cache_lock:
            if file_path in self._content_cache:
                cached_hash, content = self._content_cache[file_path]
                if cached_hash == self._get_file_hash(file_path):
                    return content
        return None

    def _cache_content(self, file_path: str, content: str):
        """Cache file content with hash."""
        with self._cache_lock:
            self._content_cache[file_path] = (self._get_file_hash(file_path), content)
            # Limit cache size
            if len(self._content_cache) > 100:
                # Remove oldest entries
                oldest = list(self._content_cache.keys())[:20]
                for key in oldest:
                    del self._content_cache[key]

    def _read_file_content(self, file_path: str) -> str:
        """Read and extract content from a file based on its type."""
        # Check cache first
        cached_content = self._get_cached_content(file_path)
        if cached_content is not None:
            return cached_content

        try:
            content = ""
            if file_path.lower().endswith('.pdf'):
                with pdfplumber.open(file_path) as pdf:
                    content = "\n".join(
                        page.extract_text() or "" 
                        for page in pdf.pages
                    )
            elif file_path.lower().endswith(('.cpp', '.h', '.java', '.py')):
                content = self._read_code_file(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

            if content:
                self._cache_content(file_path, content.lower())
                return content.lower()
            return ""
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""

    def _read_code_file(self, file_path: str) -> str:
        """Read and clean code files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Remove comments
                content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
                content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
                return content
        except Exception as e:
            print(f"Error reading code file {file_path}: {e}")
            return ""

    def _calculate_relevance_score(self, content: str, keywords: Dict[str, List[str]]) -> float:
        """Calculate relevance score based on keyword categories."""
        if not content:
            return 0
            
        content = content.lower()
        score = 0
        
        # Weight different keyword categories
        weights = {
            'language': 0.3,
            'framework': 0.3,
            'complexity': 0.1,
            'context': 0.15,
            'topic': 0.15
        }
        
        for category, words in keywords.items():
            if not words:
                continue
                
            category_score = 0
            for word in words:
                if word in content:
                    category_score += 1
            
            if words:  # Normalize by number of words in category
                category_score /= len(words)
                score += category_score * weights[category]
        
        return min(1.0, score)

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        Extract and categorize keywords from the query.
        Returns dict with categories: language, framework, complexity, context
        """
        query = query.lower()
        words = query.split()
        
        keywords = {
            'language': [],
            'framework': [],
            'complexity': [],
            'context': [],
            'topic': []
        }
        
        # Complexity indicators
        complexity_words = {
            'simple': 'basic',
            'basic': 'basic',
            'easy': 'basic',
            'advanced': 'advanced',
            'complex': 'advanced',
            'complicated': 'advanced'
        }
        
        # Context indicators
        context_words = {
            'example': 'practical',
            'program': 'practical',
            'code': 'practical',
            'tutorial': 'docs',
            'lecture': 'lectures',
            'documentation': 'docs',
            'practical': 'practical',
            'assignment': 'practical',
            'lab': 'practical'
        }
        
        # Topic indicators
        topic_words = {
            'dsa': 'data structures',
            'data structure': 'data structures',
            'algorithm': 'algorithms',
            'sorting': 'algorithms',
            'searching': 'algorithms',
            'linked list': 'data structures',
            'tree': 'data structures',
            'graph': 'data structures'
        }
        
        # Check for language mentions
        all_languages = set()
        for year_info in self.year_mapping.values():
            all_languages.update(year_info.get('languages', []))
        
        # Check for framework mentions
        all_frameworks = set()
        for mappings in self.framework_mappings.values():
            all_frameworks.update(mappings)
        
        # Process each word and common two-word phrases
        phrases = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        phrases.extend(words)
        
        for phrase in phrases:
            # Check languages
            if phrase in all_languages:
                keywords['language'].append(phrase)
            
            # Check frameworks
            if phrase in all_frameworks:
                keywords['framework'].append(phrase)
            
            # Check complexity
            if phrase in complexity_words:
                keywords['complexity'].append(complexity_words[phrase])
            
            # Check context
            if phrase in context_words:
                keywords['context'].append(context_words[phrase])
            
            # Check topics
            if phrase in topic_words:
                keywords['topic'].append(topic_words[phrase])
        
        return keywords

    def _determine_search_path(self, year_level: str, keywords: Dict[str, List[str]]) -> List[Tuple[str, str, Optional[str]]]:
        """
        Determine which folders to search based on keywords.
        Returns list of (folder_path, context_type, framework_name)
        """
        search_paths = []
        year_info = self.year_mapping.get(year_level)
        
        if not year_info:
            return search_paths
            
        base_path = os.path.join(self.resources_dir, year_info['folder'])
        
        # If a framework is mentioned, prioritize framework-specific folders
        if keywords['framework']:
            framework = keywords['framework'][0]
            for lang, frameworks in year_info.get('frameworks', {}).items():
                if framework in frameworks:
                    framework_path = os.path.join(base_path, framework.title())
                    if os.path.exists(framework_path):
                        search_paths.append((framework_path, 'framework', framework))
        
        # For Year 1 C++, handle special cases
        if year_level == "Year 1 Certificate":
            if 'practical' in keywords['context']:
                search_paths.append((os.path.join(base_path, 'Practical'), 'practical', None))
            elif any(topic in ['data structures', 'algorithms'] for topic in keywords['topic']):
                search_paths.append((os.path.join(base_path, 'C++ Docs'), 'doc', None))
            else:
                # Search all C++ folders in priority order
                for context_type in ['docs', 'practical', 'lectures']:
                    folder = year_info['contexts'][context_type]
                    path = os.path.join(base_path, folder)
                    if os.path.exists(path):
                        search_paths.append((path, context_type, None))
        
        # For other years, handle based on keywords
        else:
            # If language is specified without framework, check language-specific folders
            if keywords['language'] and not keywords['framework']:
                lang = keywords['language'][0]
                if lang == 'java' and 'data structures' in keywords['topic']:
                    dsa_path = os.path.join(base_path, 'Java DSA')
                    if os.path.exists(dsa_path):
                        search_paths.append((dsa_path, 'framework', 'dsa'))
        
        # If no specific paths found, search all folders for the year
        if not search_paths:
            for item in os.listdir(base_path):
                full_path = os.path.join(base_path, item)
                if os.path.isdir(full_path):
                    search_paths.append((full_path, 'basic', None))
        
        return search_paths

    def _process_file(self, file_data: Tuple[str, str, str, Dict]) -> Optional[ResourceMatch]:
        """Process a single file and return a ResourceMatch if relevant."""
        file_path, context_type, framework, keywords = file_data
        content = self._read_file_content(file_path)
        
        if not content:
            return None
            
        # Quick relevance check before full processing
        if not any(
            any(keyword in content for keyword in category)
            for category in keywords.values()
            if category
        ):
            return None
            
        score = self._calculate_relevance_score(content, keywords)
        if score < 0.2:  # Minimum threshold
            return None
            
        # Extract relevant sections only if we have a good score
        relevant_sections = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if any(word in line.lower() for category in keywords.values() for word in category):
                start = max(0, i - 2)  # Reduced context for speed
                end = min(len(lines), i + 3)
                section = '\n'.join(lines[start:end])
                relevant_sections.append(section)
                if len(relevant_sections) >= 3:  # Limit sections for speed
                    break
        
        if relevant_sections:
            return ResourceMatch(
                file_path=file_path,
                content='\n---\n'.join(relevant_sections),
                score=score,
                context_type=context_type,
                framework_name=framework
            )
        return None

    def search_resources(self, query: str, year_level: str, min_score: float = 0.2) -> List[ResourceMatch]:
        """
        Optimized search through resources using parallel processing.
        """
        keywords = self._extract_keywords(query)
        search_paths = self._determine_search_path(year_level, keywords)
        matches = []
        
        # Prepare file list for parallel processing
        files_to_process = []
        for base_path, context_type, framework in search_paths:
            if not os.path.exists(base_path):
                continue
                
            for root, _, files in os.walk(base_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Skip files that are likely irrelevant based on extension
                    if not file.lower().endswith(('.pdf', '.txt', '.cpp', '.h', '.java', '.py', '.md')):
                        continue
                    files_to_process.append((file_path, context_type, framework, keywords))
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(files_to_process), 4)) as executor:
            future_to_file = {
                executor.submit(self._process_file, file_data): file_data
                for file_data in files_to_process
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                try:
                    result = future.result()
                    if result:
                        matches.append(result)
                except Exception as e:
                    print(f"Error processing file: {e}")
        
        # Sort by relevance score
        matches.sort(key=lambda x: x.score, reverse=True)
        return matches[:3]  # Return top 3 matches

    def format_results(self, matches: List[ResourceMatch]) -> Optional[str]:
        """Format search results into a prompt-friendly string."""
        if not matches:
            return None
            
        formatted = "Found relevant information in local resources:\n\n"
        for match in matches:
            context_info = f" [{match.context_type}]"
            if match.framework_name:
                context_info += f" using {match.framework_name}"
                
            formatted += f"From {os.path.basename(match.file_path)}{context_info} (Relevance: {match.score:.2f}):\n"
            formatted += f"{match.content}\n\n"
            
        return formatted

    def generate_local_response(self, query: str, matches: List[ResourceMatch], year_level: str) -> Optional[str]:
        """
        Generate a response using only local resources without API calls.
        Returns None if insufficient information is found.
        """
        if not matches:
            return None

        # Check if we have enough relevant content
        total_relevance = sum(match.score for match in matches)
        if total_relevance < 0.5:  # Threshold for considering local response
            return None

        # Start building the response
        response_parts = []
        
        # Add introduction based on year level
        year_info = self.year_mapping.get(year_level, {})
        if year_info:
            response_parts.append(f"Based on the {year_level} curriculum materials, here's what I found:")
        else:
            response_parts.append("Here's what I found in our local resources:")

        # Process each match
        for match in matches:
            # Extract the most relevant parts
            content = match.content.strip()
            
            # Add context about the source
            source_info = f"\nFrom {os.path.basename(match.file_path)}"
            if match.framework_name:
                source_info += f" ({match.framework_name} framework)"
            if match.context_type:
                source_info += f" - {match.context_type}"
            
            response_parts.append(f"{source_info}:\n```\n{content}\n```")

        # Add a conclusion
        response_parts.append("\nLet me know if you need any clarification or have additional questions!")

        # Combine all parts
        final_response = "\n\n".join(response_parts)

        # Only return if the response seems substantial enough
        if len(final_response.strip()) > 100:  # Minimum response length
            return final_response
        return None