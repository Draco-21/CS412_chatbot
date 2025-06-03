import os
import json
import pdfplumber
from pathlib import Path
from typing import Dict, List, Optional
from text_cleaner import clean_documentation

RESOURCES_DIR = "resources"
CLEANED_RESOURCES_DIR = "cleaned_resources"

# Category detection keywords
CATEGORY_KEYWORDS = {
    'fund': ['basics', 'fundamental', 'introduction', 'getting started', 'syntax'],
    'ds': ['data structure', 'array', 'list', 'tree', 'graph', 'stack', 'queue'],
    'algo': ['algorithm', 'sorting', 'searching', 'recursion', 'complexity'],
    'oop': ['object', 'class', 'inheritance', 'polymorphism', 'encapsulation'],
    'web': ['html', 'css', 'javascript', 'web', 'frontend', 'backend'],
    'db': ['database', 'sql', 'query', 'table', 'join'],
    'fw': ['framework', 'library', 'api', 'spring', 'django'],
    'net': ['network', 'protocol', 'tcp', 'ip', 'socket'],
    'gen': []  # Default category
}

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """Extract text from a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def detect_category(text: str, filename: str) -> str:
    """Detect the category of the content based on keywords."""
    text_lower = text.lower()
    filename_lower = filename.lower()
    
    # Check each category's keywords
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower or keyword in filename_lower:
                return category
    
    return 'gen'  # Default to general category

def extract_code_snippets(text: str) -> List[Dict]:
    """Extract code snippets and their context from text."""
    # Split text into chunks that might contain code
    chunks = text.split('\n\n')
    code_sections = []
    
    current_code = []
    current_context = []
    
    for chunk in chunks:
        # Simple heuristic: if a chunk has common programming keywords or symbols
        if any(keyword in chunk.lower() for keyword in ['int', 'void', 'class', 'function', 'def', '#include']):
            if current_context and current_code:
                code_sections.append({
                    'context': ' '.join(current_context),
                    'code': '\n'.join(current_code)
                })
            current_code = [chunk]
            current_context = []
        else:
            if current_code:
                current_context.append(chunk)
            
    # Add the last section if exists
    if current_context and current_code:
        code_sections.append({
            'context': ' '.join(current_context),
            'code': '\n'.join(current_code)
        })
    
    return code_sections

def process_pdf(pdf_path: str) -> List[Dict]:
    """Process a single PDF file and return structured content."""
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return []
    
    # Clean the text
    cleaned_chunks = clean_documentation([text])
    if not cleaned_chunks:
        return []
    
    # Detect category
    category = detect_category(text, pdf_path)
    
    # Extract code snippets and their context
    code_sections = extract_code_snippets(text)
    
    # Create structured content
    structured_content = []
    for section in code_sections:
        content = {
            'title': os.path.basename(pdf_path).replace('.pdf', ''),
            'purpose': section['context'][:200],  # Limit context length
            'code': section['code'],
            'category': category,
            'type': 'example',
            'source_file': str(pdf_path)
        }
        structured_content.append(content)
    
    return structured_content

def process_year_directory(year_dir: str):
    """Process all PDFs in a year directory."""
    year_path = Path(RESOURCES_DIR) / year_dir
    if not year_path.exists():
        return
    
    # Create corresponding directory in cleaned_resources
    year_short = year_dir.replace('Year ', 'Y')
    cleaned_year_path = Path(CLEANED_RESOURCES_DIR) / year_short
    
    # Process all PDFs recursively
    for pdf_path in year_path.rglob('*.pdf'):
        print(f"Processing {pdf_path}...")
        structured_content = process_pdf(str(pdf_path))
        
        if structured_content:
            for content in structured_content:
                # Create category directory if needed
                category_dir = cleaned_year_path / content['category']
                category_dir.mkdir(parents=True, exist_ok=True)
                
                # Save as JSON file
                output_file = category_dir / f"{pdf_path.stem}_{content['category']}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=2)

def main():
    """Process all resources."""
    # Create cleaned_resources directory if it doesn't exist
    Path(CLEANED_RESOURCES_DIR).mkdir(exist_ok=True)
    
    # Process each year directory
    for year_dir in ['Year 1', 'Year 2', 'Year 3', 'Year 4']:
        print(f"\nProcessing {year_dir}...")
        process_year_directory(year_dir)

if __name__ == "__main__":
    main() 