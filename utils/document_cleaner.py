import os
import re
import json
import pdfplumber
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import shutil
from pathlib import Path
import hashlib

@dataclass
class CodeSection:
    title: str
    purpose: str
    code_snippet: str
    source_file: str
    year_level: str
    topic_category: str
    context_type: str
    framework_name: Optional[str] = None

class DocumentCleaner:
    def __init__(self):
        # Get the project root directory (parent of utils)
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.source_dir = os.path.join(self.project_root, "resources")
        self.output_dir = os.path.join(self.project_root, "cleaned_resources")
        
        # Updated patterns for code detection
        self.code_patterns = [
            # Traditional code blocks
            r'```(?:\w+)?\n(.*?)\n```',
            # Code snippets with common keywords
            r'(?:example|program|code|implementation|algorithm|pseudo[- ]?code)[\s]*?[:]\s*([\w\s{}\[\];()=<>"+\-*/\n.,\']+?)(?=\n\n|\Z)',
            # Function definitions (including Python)
            r'(?:def|void|int|float|double|char|bool|string|class|struct)\s+\w+\s*\([^)]*\)\s*[:{][^}]+[}]?',
            # Common programming constructs
            r'(?:for|while|if|switch)\s*\([^)]*\)\s*{[^}]+}',
            # Variable declarations and assignments
            r'(?:int|float|double|char|bool|string)\s+\w+\s*=\s*[^;]+;',
            # Python-style code blocks (indented blocks)
            r'(?:^|\n)(?:[ ]{4}|\t)[\w\s=\-+*/\[\]().,\'\"]+(?:\n(?:[ ]{4}|\t)[\w\s=\-+*/\[\]().,\'\"]+)*',
            # Machine Learning specific patterns
            r'(?:import\s+(?:numpy|tensorflow|torch|sklearn)|from\s+(?:numpy|tensorflow|torch|sklearn)\s+import)[^\n]+',
            r'(?:model|classifier|regressor)\.(?:fit|predict|transform)\([^\)]+\)',
            # Mathematical formulas and algorithms
            r'(?:Algorithm|Procedure|Function)[\s]*?[:]\s*\n(?:[ ]{2,}|\t)[\w\s=\-+*/\[\]().,\'\"]+(?:\n(?:[ ]{2,}|\t)[\w\s=\-+*/\[\]().,\'\"]+)*',
            # Numpy/Matrix operations
            r'(?:np|tf|torch)\.(?:array|matrix|zeros|ones)\([^\)]+\)',
            # Common ML variable assignments
            r'(?:X|y|theta|alpha|beta|gamma|lambda)\s*=\s*[\w\s=\-+*/\[\]().,\'\"]+',
        ]
        self.code_patterns = [re.compile(pattern, re.DOTALL | re.IGNORECASE) for pattern in self.code_patterns]
        
        # Pattern for finding section titles
        self.title_patterns = [
            r'^#+\s+(.+)$',  # Markdown style
            r'^(?:Chapter|Section)\s+\d+[.:]\s*(.+)$',  # Traditional textbook style
            r'^\d+[.:]\d*\s+(.+)$',  # Numbered sections
            r'^(?:TOPIC|LECTURE|UNIT|ALGORITHM)[:\s]+(.+)$',  # Educational content style
            r'^\s*([A-Z][A-Za-z\s]{10,})\s*$',  # Capitalized lengthy lines
            r'^(?:Definition|Theorem|Lemma|Proposition)\s+\d*[.:]\s*(.+)$',  # Mathematical content
            r'^(?:Algorithm|Procedure)[\s]*\d*[.:]\s*(.+)$'  # Algorithm titles
        ]
        self.title_patterns = [re.compile(pattern, re.MULTILINE) for pattern in self.title_patterns]
        
        # Framework mappings (matching resource_retriever.py)
        self.framework_mappings = {
            'java': ['springboot', 'spring', 'spring-boot', 'java', 'jdbc', 'hibernate'],
            'python': ['flask', 'django', 'python', 'numpy', 'pandas', 'tensorflow', 'pytorch', 'scikit-learn', 'sklearn'],
            'web': ['asp.net', 'asp', '.net', 'flask', 'html', 'css', 'javascript', 'js', 'react', 'angular'],
            'network': ['socket', 'cisco', 'tcp/ip', 'http', 'networking'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'database'],
            'machine_learning': ['tensorflow', 'pytorch', 'sklearn', 'keras', 'torch', 'ml', 'ai', 
                               'neural network', 'deep learning', 'machine learning', 'numpy', 'pandas']
        }
        
        # Context type patterns
        self.context_patterns = {
            'framework': r'framework|library|module|package|api',
            'basic': r'basic|fundamental|introduction|concept|overview',
            'lecture': r'lecture|lesson|tutorial|guide|notes',
            'practical': r'practical|exercise|assignment|lab|workshop|example|implementation',
            'algorithm': r'algorithm|procedure|pseudocode|steps|method',
            'math': r'theorem|proof|lemma|definition|proposition',
            'doc': r'documentation|manual|reference|guide'
        }
        
    def setup_output_directory(self):
        """Create the output directory structure."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        
        # Create year level directories with shorter names
        for year in ["Year 1", "Year 2", "Year 3", "Year 4"]:
            year_short = self.get_short_year(year)
            # Create main categories with shorter names
            categories = [
                "Fundamentals",
                "DataStructures",
                "Algorithms",
                "OOP",
                "Frameworks",
                "Networking",
                "Database",
                "WebDev",
                "General"
            ]
            
            for category in categories:
                cat_short = self.get_short_category(category)
                category_path = os.path.join(self.output_dir, year_short, cat_short)
                os.makedirs(category_path, exist_ok=True)
                
                # Create framework-specific subdirectories in relevant categories
                if category in ["Frameworks", "WebDev"]:
                    for lang, frameworks in self.framework_mappings.items():
                        for framework in frameworks:
                            # Use shortened framework names
                            fw_short = framework[:4].lower()
                            os.makedirs(os.path.join(category_path, fw_short), exist_ok=True)

    def extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks using multiple patterns."""
        code_blocks = []
        for pattern in self.code_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                code = match.group(1) if len(match.groups()) > 0 else match.group(0)
                # Basic validation to ensure it looks like code
                if len(code.strip()) > 0 and any(keyword in code.lower() for keyword in [
                    'for', 'while', 'if', 'else', 'class', 'def', 'function', 'return', 'int', 'float'
                ]):
                    code_blocks.append(code.strip())
        return code_blocks

    def extract_section_title(self, text: str) -> str:
        """Extract section title using multiple patterns."""
        for pattern in self.title_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return "Untitled Section"

    def determine_framework(self, content: str) -> Optional[str]:
        """Determine if content is related to a specific framework."""
        content_lower = content.lower()
        for lang, frameworks in self.framework_mappings.items():
            for framework in frameworks:
                if framework in content_lower:
                    return framework
        return None

    def determine_context_type(self, content: str) -> str:
        """Determine the type of content (framework, basic, lecture, etc.)."""
        content_lower = content.lower()
        for context_type, pattern in self.context_patterns.items():
            if re.search(pattern, content_lower):
                return context_type
        return 'doc'  # Default to documentation

    def determine_topic_category(self, title: str, content: str) -> str:
        """Determine the topic category based on content analysis."""
        title_lower = title.lower()
        content_lower = content.lower()
        
        categories = {
            "Fundamentals": [
                "basic", "fundamental", "syntax", "variable", "loop", "condition",
                "function", "operator", "control flow"
            ],
            "DataStructures": [
                "array", "list", "stack", "queue", "tree", "graph", "hash",
                "linked list", "dictionary", "set", "matrix", "vector"
            ],
            "Algorithms": [
                "algorithm", "sort", "search", "recursive", "iteration",
                "complexity", "optimization", "dynamic programming",
                "gradient descent", "backpropagation", "perceptron"
            ],
            "MachineLearning": [
                "machine learning", "neural network", "deep learning", "training",
                "prediction", "classification", "regression", "clustering",
                "supervised", "unsupervised", "reinforcement", "validation",
                "overfitting", "regularization", "gradient descent", "backpropagation"
            ],
            "Mathematics": [
                "theorem", "proof", "lemma", "proposition", "derivative",
                "gradient", "vector", "matrix", "probability", "statistics",
                "expected value", "distribution", "linear algebra"
            ],
            "OOP": [
                "class", "object", "inheritance", "polymorphism", "encapsulation",
                "interface", "abstract", "method"
            ],
            "Frameworks": list(sum([frameworks for frameworks in self.framework_mappings.values()], [])),
            "Networking": [
                "network", "socket", "protocol", "tcp", "udp", "http", "api",
                "request", "response"
            ],
            "Database": [
                "database", "sql", "query", "table", "join", "index", "crud",
                "transaction"
            ],
            "WebDev": [
                "html", "css", "javascript", "frontend", "backend", "api",
                "web", "rest", "http"
            ]
        }
        
        # Check content against categories
        for category, keywords in categories.items():
            if any(keyword in title_lower or keyword in content_lower for keyword in keywords):
                return category
        return "General"

    def extract_purpose(self, text: str, max_length: int = 300) -> str:
        """Extract the purpose/description from the text."""
        # Look for purpose indicators
        purpose_indicators = [
            r'(?:purpose|objective|goal|aim|description)[:]\s*([^.!?\n]+[.!?\n])',
            r'(?:this section|this chapter|this topic)\s+(?:covers|describes|explains|introduces)\s+([^.!?\n]+[.!?\n])',
            r'(?:in this|this)\s+(?:section|chapter|topic)\s*[,.]\s*([^.!?\n]+[.!?\n])'
        ]
        
        for pattern in purpose_indicators:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no clear purpose found, take the first non-empty paragraph
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if para.strip() and not any(para.strip().startswith(x) for x in ['#', '```', 'Chapter', 'Section']):
                clean_para = ' '.join(para.split())
                return clean_para[:max_length] + ('...' if len(clean_para) > max_length else '')
        
        return "No purpose provided"

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize the filename to remove invalid characters and make it Windows-compatible.
        Also ensures the filename is short enough to avoid path length issues.
        """
        # Replace invalid characters with underscores
        invalid_chars = r'[<>:"/\\|?*]'
        # First replace spaces and invalid chars with underscores
        safe_name = re.sub(invalid_chars, '_', filename)
        safe_name = re.sub(r'\s+', '_', safe_name)
        
        # Remove any dots at the end (Windows restriction)
        safe_name = safe_name.rstrip('.')
        
        # Take only the first few words for the filename
        words = safe_name.split('_')
        if len(words) > 3:
            safe_name = '_'.join(words[:3])
        
        # Ensure the filename isn't too long
        if len(safe_name) > 50:  # Much shorter to avoid path length issues
            safe_name = safe_name[:47] + "..."
            
        # Remove any remaining problematic characters
        safe_name = ''.join(c for c in safe_name if c.isprintable())
        
        # Ensure we don't have consecutive underscores
        safe_name = re.sub(r'_+', '_', safe_name)
        
        return safe_name.lower()  # Convert to lowercase for consistency

    def get_short_year(self, year_level: str) -> str:
        """Convert year level to short form."""
        return f"Y{year_level.split()[1]}"  # "Year 1" -> "Y1"

    def get_short_category(self, category: str) -> str:
        """Convert category to short form."""
        category_map = {
            "Fundamentals": "fund",
            "DataStructures": "ds",
            "Algorithms": "algo",
            "MachineLearning": "ml",
            "Mathematics": "math",
            "OOP": "oop",
            "Frameworks": "fw",
            "Networking": "net",
            "Database": "db",
            "WebDev": "web",
            "General": "gen"
        }
        return category_map.get(category, category.lower()[:4])

    def save_section(self, section: CodeSection):
        """Save a code section to the appropriate directory."""
        try:
            # Use shorter directory names
            year_short = self.get_short_year(section.year_level)
            cat_short = self.get_short_category(section.topic_category)
            
            # Sanitize the title for use in filename
            safe_title = self.sanitize_filename(section.title)
            
            # Create a unique identifier for this section
            section_id = hashlib.md5(
                f"{section.title}{section.code_snippet[:100]}".encode()
            ).hexdigest()[:8]
            
            # Create a more compact directory structure
            if section.framework_name:
                # For framework-specific content
                base_dir = os.path.join(self.output_dir, year_short, cat_short, section.framework_name[:4])
            else:
                # For general content
                base_dir = os.path.join(self.output_dir, year_short, cat_short)
            
            # Create the output filename
            output_path = os.path.join(base_dir, f"{safe_title}_{section_id}.json")
            
            # Ensure the total path length is within Windows limits
            if len(output_path) >= 250:  # Windows MAX_PATH is 260
                # Further shorten the filename if needed
                base_path = os.path.dirname(output_path)
                ext = os.path.splitext(output_path)[1]
                available_length = 240 - len(base_path) - len(ext)  # Leave some margin
                short_name = f"{safe_title[:available_length-10]}_{section_id}{ext}"
                output_path = os.path.join(base_path, short_name)
            
            data = {
                "title": section.title,
                "purpose": section.purpose,
                "code_snippet": section.code_snippet,
                "source_file": section.source_file,
                "year_level": section.year_level,
                "topic_category": section.topic_category,
                "context_type": section.context_type,
                "framework_name": section.framework_name
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving section '{section.title}': {str(e)}")
            print(f"Attempted path: {output_path if 'output_path' in locals() else 'path not created'}")

    def process_pdf(self, file_path: str, year_level: str) -> List[CodeSection]:
        """Process a PDF file and extract code sections."""
        sections = []
        try:
            with pdfplumber.open(file_path) as pdf:
                current_text = ""
                for page in pdf.pages:
                    current_text += page.extract_text() + "\n"
                
                # Split into sections (using various section markers)
                raw_sections = []
                current_section = []
                
                for line in current_text.split('\n'):
                    # Check if this line could be a section title
                    is_title = any(pattern.match(line.strip()) for pattern in self.title_patterns)
                    
                    if is_title and current_section:
                        raw_sections.append('\n'.join(current_section))
                        current_section = []
                    
                    current_section.append(line)
                
                if current_section:
                    raw_sections.append('\n'.join(current_section))
                
                # If no sections found, treat the whole document as one section
                if not raw_sections:
                    raw_sections = [current_text]
                
                for section in raw_sections:
                    if not section.strip():
                        continue
                    
                    title = self.extract_section_title(section)
                    code_blocks = self.extract_code_blocks(section)
                    
                    if code_blocks:
                        purpose = self.extract_purpose(section)
                        category = self.determine_topic_category(title, section)
                        context_type = self.determine_context_type(section)
                        framework = self.determine_framework(section)
                        
                        for i, code in enumerate(code_blocks):
                            # Add a number suffix if multiple code blocks in same section
                            section_title = f"{title}_part{i+1}" if len(code_blocks) > 1 else title
                            
                            sections.append(CodeSection(
                                title=section_title,
                                purpose=purpose,
                                code_snippet=code.strip(),
                                source_file=os.path.basename(file_path),
                                year_level=year_level,
                                topic_category=category,
                                context_type=context_type,
                                framework_name=framework
                            ))
                            
                            print(f"Found code section in {os.path.basename(file_path)}:")
                            print(f"  Title: {section_title}")
                            print(f"  Category: {category}")
                            print(f"  Context: {context_type}")
                            if framework:
                                print(f"  Framework: {framework}")
                            print("  Code length:", len(code.strip()), "characters")
                            print()
                            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
        
        return sections

    def process_directory(self, progress_callback=None):
        """Process all files in the source directory."""
        # Only process Year 4
        year_dir = "Year 4"
        year_path = os.path.join(self.source_dir, year_dir)
        
        if not os.path.exists(year_path):
            print(f"Error: Year 4 directory not found at {year_path}")
            return
            
        # Create only Year 4 directory structure
        year_short = self.get_short_year(year_dir)
        categories = [
            "Fundamentals",
            "DataStructures",
            "Algorithms",
            "MachineLearning",
            "Mathematics",
            "OOP",
            "Frameworks",
            "Networking",
            "Database",
            "WebDev",
            "General"
        ]
        
        # Create base directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create Year 4 category directories
        for category in categories:
            cat_short = self.get_short_category(category)
            category_path = os.path.join(self.output_dir, year_short, cat_short)
            os.makedirs(category_path, exist_ok=True)
            
            # Create framework-specific subdirectories in relevant categories
            if category in ["Frameworks", "WebDev", "MachineLearning"]:
                for lang, frameworks in self.framework_mappings.items():
                    for framework in frameworks:
                        # Use shortened framework names
                        fw_short = framework[:4].lower()
                        os.makedirs(os.path.join(category_path, fw_short), exist_ok=True)
        
        # Process all PDF files in Year 4
        print(f"\nProcessing Year 4 content from: {year_path}")
        for root, _, files in os.walk(year_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    print(f"\nProcessing: {file_path}")
                    
                    sections = self.process_pdf(file_path, year_dir)
                    if sections:
                        print(f"Found {len(sections)} sections in {file}")
                        for section in sections:
                            self.save_section(section)
                    else:
                        print(f"No code sections found in {file}")
                    
                    if progress_callback:
                        progress_callback()

def main():
    cleaner = DocumentCleaner()
    cleaner.process_directory()

if __name__ == "__main__":
    main() 