# vector_retriever.py
import os
import time # <--- Add this import
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from text_extractor import extract_text
import pickle

RESOURCES_BASE_FOLDER = "resources"
VECTOR_INDEX_FILE = "vector_index.faiss"
VECTOR_MAPPING_FILE = "vector_mapping.pkl"
MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

YEAR_FOLDER_MAP = {
    "Year 1 Certificate": "Year 1",
    "Year 2 Diploma": "Year 2",
    "Year 3 Degree": "Year 3",
    "Year 4 Postgraduate Diploma": "Year 4"
}

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    start = 0
    chunks = []
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += (size - overlap)
    return chunks

def create_vector_index() -> tuple[bool, float]: # <--- Modified return type
    """Builds the FAISS vector index."""
    start_time = time.time() # <--- Record start time

    print("VectorRetriever: Loading sentence transformer model...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"VectorRetriever: Error loading SentenceTransformer model: {e}")
        print("VectorRetriever: Make sure you have an internet connection if downloading the model for the first time,")
        print("VectorRetriever: and that 'sentence-transformers' and PyTorch are correctly installed.")
        return False, 0.0
    print("VectorRetriever: Model loaded.")

    all_chunks = []
    mapping = [] # Stores {'path': file_path, 'year': year_folder, 'chunk_text': chunk}
    print("VectorRetriever: Starting to process and chunk resources...")

    for year_str, year_folder_short_name in YEAR_FOLDER_MAP.items():
        year_path = os.path.join(RESOURCES_BASE_FOLDER, year_folder_short_name)
        if os.path.isdir(year_path):
            print(f"VectorRetriever: Processing folder: {year_path}")
            for root, _, files in os.walk(year_path):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    content = extract_text(file_path)
                    if content:
                        doc_chunks = chunk_text(content)
                        for chunk in doc_chunks:
                            all_chunks.append(chunk) # Store chunk text for embedding
                            mapping.append({'path': file_path, 'year': year_folder_short_name, 'chunk_text': chunk})
        else:
            print(f"VectorRetriever: Year folder not found: {year_path}")


    if not all_chunks:
        print("VectorRetriever: No text found to index.")
        return False, 0.0

    print(f"VectorRetriever: Found {len(all_chunks)} chunks. Generating embeddings (this may take a while)...")
    try:
        embeddings = model.encode(all_chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
    except Exception as e:
        print(f"VectorRetriever: Error generating embeddings: {e}")
        return False, 0.0
        
    dimension = embeddings.shape[1]

    print(f"VectorRetriever: Embeddings generated. Dimension: {dimension}. Building FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"VectorRetriever: FAISS index built. Saving index and mapping...")
    try:
        faiss.write_index(index, VECTOR_INDEX_FILE)
        with open(VECTOR_MAPPING_FILE, 'wb') as f:
            pickle.dump(mapping, f) # Save the mapping of index to chunk details
    except Exception as e:
        print(f"VectorRetriever: Error saving index or mapping: {e}")
        return False, 0.0

    end_time = time.time() # <--- Record end time
    duration = end_time - start_time # <--- Calculate duration
    print(f"VectorRetriever: Vector indexing complete. Indexed {len(all_chunks)} chunks in {duration:.2f} seconds.")
    return True, duration # <--- Return success and duration

# --- search_vector_index function remains the same ---
def search_vector_index(query_str: str, year_level_filter: str | None = None, k=1) -> str | None: # Default k=1 to get one best chunk
    if not os.path.exists(VECTOR_INDEX_FILE) or not os.path.exists(VECTOR_MAPPING_FILE):
        print("VectorRetriever: Index or mapping file not found. Please create the index first.")
        return None
    try:
        index = faiss.read_index(VECTOR_INDEX_FILE)
        with open(VECTOR_MAPPING_FILE, 'rb') as f:
            mapping = pickle.load(f)
        
        model = SentenceTransformer(MODEL_NAME) # Load model for query embedding
        query_vector = model.encode([query_str]).astype('float32')

        print(f"VectorRetriever: Searching for query: {query_str}")
        distances, indices = index.search(query_vector, k)

        if len(indices[0]) == 0 or indices[0][0] < 0 : # Check if any valid index found
            print(f"VectorRetriever: No results found for query: {query_str}")
            return None

        # For now, return the best matching chunk.
        # Year level filtering can be added here by checking mapping[indices[0][i]]['year']
        best_chunk_index = indices[0][0]
        best_hit_details = mapping[best_chunk_index]

        print(f"VectorRetriever: Best match found in '{best_hit_details['path']}' (Distance: {distances[0][0]:.4f})")
        return best_hit_details['chunk_text'] # Return the text of the best chunk

    except Exception as e:
        print(f"VectorRetriever: Error during vector search: {e}")
        return None