import re
import string
from typing import List, Set
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text by removing noise and standardizing format.
    """
    if not text or not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove special characters and numbers but keep meaningful punctuation
    text = re.sub(r'[^a-zA-Z\s\.\,\?\!]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def is_meaningful_content(text: str, min_words: int = 5, min_unique_words: int = 3) -> bool:
    """
    Check if the text contains meaningful content based on various criteria.
    """
    if not text:
        return False
        
    # Tokenize words
    words = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    words = [word for word in words if word not in stop_words and word not in string.punctuation]
    
    # Check minimum word count
    if len(words) < min_words:
        return False
        
    # Check minimum unique words
    if len(set(words)) < min_unique_words:
        return False
        
    return True

def filter_technical_content(text: str) -> bool:
    """
    Check if the text contains technical/programming-related content.
    """
    # Technical keywords to look for
    technical_keywords = {
        'programming', 'code', 'function', 'class', 'method',
        'algorithm', 'data', 'structure', 'variable', 'loop',
        'database', 'api', 'framework', 'library', 'software',
        'development', 'testing', 'debug', 'implementation',
        'interface', 'system', 'application', 'web', 'network',
        'security', 'design', 'pattern', 'architecture'
    }
    
    # Tokenize and lemmatize words
    words = word_tokenize(text.lower())
    lemmatized_words = {lemmatizer.lemmatize(word) for word in words}
    
    # Check for technical keyword presence
    technical_word_count = len(technical_keywords.intersection(lemmatized_words))
    
    # Return True if there are enough technical words
    return technical_word_count >= 1

def remove_duplicates(chunks: List[str], similarity_threshold: float = 0.8) -> List[str]:
    """
    Remove duplicate or near-duplicate chunks using TF-IDF and cosine similarity.
    """
    if not chunks:
        return []
        
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(chunks)
    
    # Calculate pairwise similarities
    similarities = cosine_similarity(tfidf_matrix)
    
    # Keep track of chunks to remove
    to_remove = set()
    
    # Find duplicates
    for i in range(len(chunks)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(chunks)):
            if j in to_remove:
                continue
            if similarities[i, j] > similarity_threshold:
                # Keep the longer chunk
                if len(chunks[i]) >= len(chunks[j]):
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break
    
    # Return filtered chunks
    return [chunk for i, chunk in enumerate(chunks) if i not in to_remove]

def clean_chunk(chunk: str) -> str:
    """
    Clean a single chunk of text.
    """
    # Basic preprocessing
    chunk = preprocess_text(chunk)
    
    # Split into sentences
    sentences = sent_tokenize(chunk)
    
    # Filter meaningful sentences
    meaningful_sentences = [
        sent for sent in sentences
        if is_meaningful_content(sent) and filter_technical_content(sent)
    ]
    
    # Rejoin filtered sentences
    return ' '.join(meaningful_sentences)

def clean_documentation(chunks: List[str]) -> List[str]:
    """
    Main function to clean documentation chunks.
    """
    # Clean each chunk
    cleaned_chunks = [clean_chunk(chunk) for chunk in chunks]
    
    # Remove empty chunks
    cleaned_chunks = [chunk for chunk in cleaned_chunks if chunk.strip()]
    
    # Remove duplicates
    cleaned_chunks = remove_duplicates(cleaned_chunks)
    
    return cleaned_chunks 