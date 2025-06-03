import os
import re
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pdfplumber
import docx
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import traceback

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('resource_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('wordnet')
except LookupError:
    logger.info("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

@dataclass
class CodeSnippet:
    code: str
    language: str
    context: str
    purpose: str
    source_file: str
    line_numbers: Optional[Tuple[int, int]] = None

class ResourceProcessor:
    def __init__(self):
        self.RESOURCES_DIR = "resources"
        self.CLEANED_RESOURCES_DIR = "cleaned_resources"
        
        # Topic/Category keywords for classification
        self.TOPIC_KEYWORDS = {
            'flask': ['flask', 'web framework', 'route', 'template', 'jinja'],
            'django': ['django', 'model', 'view', 'template', 'orm'],
            'data_structures': ['array', 'list', 'tree', 'graph', 'stack', 'queue', 'hash', 'heap'],
            'algorithms': ['sort', 'search', 'recursion', 'dynamic programming', 'greedy'],
            'cpp': ['c++', 'cpp', 'vector', 'iostream', 'class'],
            'python': ['python', 'def', 'class', 'import', 'list', 'dict'],
            'java': ['java', 'class', 'public', 'private', 'interface'],
            'web_dev': ['html', 'css', 'javascript', 'dom', 'api'],
            'database': ['sql', 'database', 'query', 'table', 'join'],
            'networking': ['tcp', 'ip', 'socket', 'protocol', 'http'],
            'machine_learning': ['ml', 'neural', 'training', 'model', 'prediction'],
            'fundamentals': ['basic', 'syntax', 'variable', 'loop', 'condition']
        }
        
        # Programming language detection patterns
        self.LANGUAGE_PATTERNS = {
            'python': [r'\.py$', r'def\s+\w+\s*\(', r'import\s+\w+', r'from\s+\w+\s+import'],
            'cpp': [r'\.cpp$', r'\.h$', r'#include', r'using\s+namespace', r'int\s+main\s*\('],
            'java': [r'\.java$', r'public\s+class', r'private\s+\w+', r'System\.out'],
            'javascript': [r'\.js$', r'function\s+\w+\s*\(', r'const\s+\w+', r'let\s+\w+'],
            'html': [r'\.html$', r'<html>', r'<body>', r'<div'],
            'css': [r'\.css$', r'{', r'margin:', r'padding:'],
            'sql': [r'SELECT\s+.*\s+FROM', r'INSERT\s+INTO', r'CREATE\s+TABLE']
        }
        
        # Year-specific topics
        self.YEAR_TOPICS = {
            'Y1': ['fundamentals', 'cpp', 'python', 'data_structures'],
            'Y2': ['algorithms', 'web_dev', 'database', 'java'],
            'Y3': ['flask', 'django', 'networking', 'database'],
            'Y4': ['machine_learning', 'algorithms', 'networking']
        }
        
        logger.info(f"ResourceProcessor initialized with resources dir: {self.RESOURCES_DIR}")
        logger.info(f"Cleaned resources will be saved to: {self.CLEANED_RESOURCES_DIR}")

    def detect_programming_language(self, text: str, filename: str) -> str:
        """Detect the programming language from text content and filename."""
        text_lower = text.lower()
        
        # First check filename extension
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            if any(re.search(pattern, filename, re.IGNORECASE) for pattern in patterns):
                logger.debug(f"Language {lang} detected from filename: {filename}")
                return lang
        
        # Then check content patterns
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                logger.debug(f"Language {lang} detected from content patterns")
                return lang
        
        return 'unknown'

    def detect_topics(self, text: str, filename: str) -> List[str]:
        """Detect relevant topics from the content."""
        text_lower = text.lower()
        detected_topics = set()
        
        # Enhanced topic keywords for educational content
        topic_indicators = {
            'fundamentals': [
                'basic', 'syntax', 'variable', 'loop', 'condition',
                'introduction', 'getting started', 'tutorial', 'learn'
            ],
            'data_structures': [
                'array', 'list', 'tree', 'graph', 'stack', 'queue',
                'linked list', 'binary tree', 'hash table', 'heap'
            ],
            'algorithms': [
                'sort', 'search', 'recursion', 'complexity',
                'algorithm', 'binary search', 'merge sort', 'quick sort'
            ],
            'web_dev': [
                'html', 'css', 'javascript', 'api',
                'web', 'frontend', 'backend', 'http'
            ],
            'database': [
                'sql', 'query', 'table', 'database',
                'join', 'select', 'insert', 'update'
            ],
            'networking': [
                'tcp', 'ip', 'socket', 'protocol',
                'network', 'client', 'server', 'packet'
            ],
            'machine_learning': [
                'ml', 'neural', 'training', 'model',
                'learning', 'prediction', 'classification'
            ]
        }
        
        # Check filename first
        for topic, keywords in topic_indicators.items():
            if any(keyword in filename.lower() for keyword in keywords):
                detected_topics.add(topic)
        
        # Check content
        for topic, keywords in topic_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.add(topic)
        
        # If no topics detected, use 'fundamentals' as default
        if not detected_topics:
            detected_topics.add('fundamentals')
        
        return list(detected_topics)

    def extract_code_snippets(self, text: str, language: str) -> List[CodeSnippet]:
        """Extract code snippets and their context from text."""
        snippets = []
        lines = text.split('\n')
        
        in_code_block = False
        current_code = []
        current_context = []
        start_line = 0
        
        # Common code indicators in educational content
        code_start_indicators = [
            '```', '/*', '*/', '//', '#',
            'Example:', 'Code:', 'Program:',
            '#include', 'using namespace',
            'int main', 'void main',
            'class ', 'struct ',
            'def ', 'function'
        ]
        
        # Improved code block detection
        for i, line in enumerate(lines, 1):
            line_lower = line.lower().strip()
            
            # Detect code block start
            is_code_start = any(indicator.lower() in line_lower for indicator in code_start_indicators)
            has_code_markers = any(char in line for char in ['{', '}', ';', '()', '[]'])
            
            if (is_code_start or has_code_markers) and not in_code_block:
                in_code_block = True
                start_line = i
                if current_context:
                    current_context = current_context[-3:]  # Keep last 3 context lines
            elif in_code_block and line.strip() == '':
                # Empty line might end a code block
                consecutive_empty = 0
                for next_line in lines[i:min(i+3, len(lines))]:
                    if next_line.strip() == '':
                        consecutive_empty += 1
                    else:
                        break
                if consecutive_empty >= 2:  # Two or more empty lines end the block
                    in_code_block = False
                    if current_code:
                        snippet = CodeSnippet(
                            code='\n'.join(current_code),
                            language=self._detect_language('\n'.join(current_code), language),
                            context='\n'.join(current_context),
                            purpose=self._extract_purpose(current_context),
                            source_file='',
                            line_numbers=(start_line, i)
                        )
                        snippets.append(snippet)
                    current_code = []
                    current_context = []
            
            if in_code_block:
                current_code.append(line)
            else:
                current_context.append(line)
        
        # Handle any remaining code block
        if current_code:
            snippet = CodeSnippet(
                code='\n'.join(current_code),
                language=self._detect_language('\n'.join(current_code), language),
                context='\n'.join(current_context),
                purpose=self._extract_purpose(current_context),
                source_file='',
                line_numbers=(start_line, len(lines))
            )
            snippets.append(snippet)
        
        logger.debug(f"Extracted {len(snippets)} code snippets")
        return snippets

    def _detect_language(self, code: str, default_language: str) -> str:
        """Detect programming language from code content."""
        code_lower = code.lower()
        
        # C++ indicators
        if any(indicator in code_lower for indicator in 
               ['#include', 'using namespace', 'cout', 'cin', 'vector<', 
                'int main()', 'void main()', '::', '->']):
            return 'cpp'
        
        # Python indicators
        elif any(indicator in code_lower for indicator in 
                ['def ', 'class ', 'import ', 'print(', '__init__', 
                 'self.', 'if __name__']):
            return 'python'
        
        # Java indicators
        elif any(indicator in code_lower for indicator in 
                ['public class', 'private ', 'protected ', 'System.out',
                 'String[] args', '@Override']):
            return 'java'
        
        # JavaScript indicators
        elif any(indicator in code_lower for indicator in 
                ['function ', 'const ', 'let ', 'var ', 'document.',
                 '=>', 'console.log']):
            return 'javascript'
        
        return default_language

    def _extract_purpose(self, context_lines: List[str]) -> str:
        """Extract the purpose description from context lines."""
        try:
            # Join context lines and split into sentences
            text = ' '.join(context_lines)
            sentences = sent_tokenize(text)
            
            # Look for purpose-indicating sentences
            purpose_indicators = ['purpose', 'goal', 'aims to', 'designed to', 'implements', 'demonstrates']
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in purpose_indicators):
                    return sentence
            
            # If no clear purpose found, return the first sentence
            return sentences[0] if sentences else ""
        except Exception as e:
            logger.warning(f"Error extracting purpose: {str(e)}")
            return ""

    def process_file(self, file_path: str) -> List[Dict]:
        """Process a single file and return structured content."""
        logger.info(f"Processing file: {file_path}")
        
        try:
            # Extract text based on file type
            text = ""
            if file_path.endswith('.pdf'):
                try:
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            text += page.extract_text() + "\n"
                except Exception as e:
                    logger.error(f"Error reading PDF {file_path}: {str(e)}")
                    return []
            elif file_path.endswith('.docx'):
                try:
                    doc = docx.Document(file_path)
                    text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
                except Exception as e:
                    logger.error(f"Error reading DOCX {file_path}: {str(e)}")
                    return []
            else:  # Text or code files
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}")
                    return []

            if not text.strip():
                logger.warning(f"No text content found in {file_path}")
                return []

            # Detect language and topics
            language = self.detect_programming_language(text, file_path)
            topics = self.detect_topics(text, file_path)
            
            # Extract code snippets
            snippets = self.extract_code_snippets(text, language)
            
            # Create structured content
            structured_content = []
            for snippet in snippets:
                snippet.source_file = str(file_path)
                content = {
                    'title': Path(file_path).stem,
                    'language': snippet.language,
                    'topics': topics,
                    'purpose': snippet.purpose,
                    'code': snippet.code,
                    'context': snippet.context,
                    'source_file': snippet.source_file,
                    'line_numbers': snippet.line_numbers
                }
                structured_content.append(content)

            logger.info(f"Successfully processed {file_path}, found {len(structured_content)} content items")
            return structured_content

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def determine_year_level(self, file_path: str) -> str:
        """Determine the year level based on file path and content."""
        path_str = str(file_path).lower()
        
        # Check path for year indicators
        for year in ['year 1', 'year 2', 'year 3', 'year 4']:
            if year in path_str:
                logger.debug(f"Detected year level from path: {year}")
                return f"Y{year[-1]}"
        
        logger.debug(f"No year level found in path {file_path}, defaulting to Y1")
        return "Y1"

    def process_resources(self):
        """Process all resources and organize them into the cleaned_resources directory."""
        logger.info("Starting resource processing...")
        
        try:
            # Create cleaned_resources directory
            cleaned_root = Path(self.CLEANED_RESOURCES_DIR)
            if cleaned_root.exists():
                shutil.rmtree(cleaned_root)
            cleaned_root.mkdir()
            logger.info(f"Created clean directory: {cleaned_root}")
            
            # Create year directories
            for year in ['Y1', 'Y2', 'Y3', 'Y4']:
                (cleaned_root / year).mkdir()
            logger.info("Created year directories")
            
            # Process all files
            resources_root = Path(self.RESOURCES_DIR)
            if not resources_root.exists():
                logger.error(f"Resources directory not found: {resources_root}")
                return
            
            processed_files = 0
            empty_dirs = set()
            
            # Process files
            for file_path in resources_root.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.txt', '.py', '.cpp', '.h', '.java', '.js']:
                    try:
                        # Process file
                        content_list = self.process_file(str(file_path))
                        if not content_list:
                            continue
                        
                        # Determine year level
                        year_level = self.determine_year_level(file_path)
                        
                        # Save content for each detected topic
                        for content in content_list:
                            for topic in content['topics']:
                                # Create topic directory if it's relevant for the year
                                if topic in self.YEAR_TOPICS.get(year_level, []):
                                    topic_dir = cleaned_root / year_level / topic
                                    topic_dir.mkdir(exist_ok=True)
                                    
                                    # Save as JSON
                                    output_file = topic_dir / f"{file_path.stem}_{topic}.json"
                                    with open(output_file, 'w', encoding='utf-8') as f:
                                        json.dump(content, f, indent=2)
                                    
                                    processed_files += 1
                                    logger.info(f"Saved content to {output_file}")
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
                        logger.error(traceback.format_exc())
                        continue
            
            # Remove empty directories
            for year_dir in cleaned_root.iterdir():
                if year_dir.is_dir():
                    for topic_dir in year_dir.iterdir():
                        if topic_dir.is_dir() and not any(topic_dir.iterdir()):
                            empty_dirs.add(topic_dir)
            
            for empty_dir in empty_dirs:
                empty_dir.rmdir()
                logger.info(f"Removed empty directory: {empty_dir}")
            
            logger.info(f"Processing complete. Processed {processed_files} files.")
            
        except Exception as e:
            logger.error(f"Error during resource processing: {str(e)}")
            logger.error(traceback.format_exc())

def main():
    try:
        processor = ResourceProcessor()
        processor.process_resources()
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 