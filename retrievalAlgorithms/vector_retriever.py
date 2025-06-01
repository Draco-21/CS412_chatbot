# vector_retriever.py
import os
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from retrievalAlgorithms.text_extractor import extract_text, chunk_text # Import chunk_text
import pickle

RESOURCES_BASE_FOLDER = "resources"
VECTOR_INDEX_FILE = "vector_index.faiss"
VECTOR_MAPPING_FILE = "vector_mapping.pkl" # Stores list of {'path': file_path, 'year': year_folder_short_name, 'chunk_text': chunk}
MODEL_NAME = 'all-MiniLM-L6-v2' # Efficient and good quality

YEAR_FOLDER_MAP = {
    "Year 1 Certificate": "Year 1",
    "Year 2 Diploma": "Year 2",
    "Year 3 Degree": "Year 3",
    "Year 4 Postgraduate Diploma": "Year 4"
}

def create_vector_index() -> tuple[bool, float]:
    start_time = time.time()
    print("VectorRetriever: Loading sentence transformer model...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"VectorRetriever: Error loading SentenceTransformer model: {e}")
        return False, 0.0
    print("VectorRetriever: Model loaded.")

    all_chunk_texts_for_embedding = []
    mapping_data = [] # This will store {'path': ..., 'year': ..., 'chunk_text': ...} for each chunk
    
    print("VectorRetriever: Starting to process and chunk resources...")
    for year_str, year_folder_short_name in YEAR_FOLDER_MAP.items():
        year_path = os.path.join(RESOURCES_BASE_FOLDER, year_folder_short_name)
        if os.path.isdir(year_path):
            print(f"VectorRetriever: Processing folder: {year_path}")
            for root, _, files in os.walk(year_path):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    full_content = extract_text(file_path)
                    if full_content:
                        doc_chunks = chunk_text(full_content)
                        for chunk_content in doc_chunks:
                            all_chunk_texts_for_embedding.append(chunk_content)
                            mapping_data.append({
                                'path': file_path,
                                'year': year_folder_short_name,
                                'chunk_text': chunk_content # Store the actual chunk text
                            })
        else:
            print(f"VectorRetriever: Year folder not found: {year_path}")

    if not all_chunk_texts_for_embedding:
        print("VectorRetriever: No text found to index.")
        return False, 0.0

    print(f"VectorRetriever: Found {len(all_chunk_texts_for_embedding)} chunks. Generating embeddings...")
    try:
        embeddings = model.encode(all_chunk_texts_for_embedding, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
    except Exception as e:
        print(f"VectorRetriever: Error generating embeddings: {e}")
        return False, 0.0
        
    dimension = embeddings.shape[1]
    print(f"VectorRetriever: Embeddings generated. Dimension: {dimension}. Building FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"VectorRetriever: FAISS index built. Saving index and mapping data...")
    try:
        faiss.write_index(index, VECTOR_INDEX_FILE)
        with open(VECTOR_MAPPING_FILE, 'wb') as f:
            pickle.dump(mapping_data, f)
    except Exception as e:
        print(f"VectorRetriever: Error saving index or mapping: {e}")
        return False, 0.0

    duration = time.time() - start_time
    print(f"VectorRetriever: Vector indexing complete. Indexed {len(all_chunk_texts_for_embedding)} chunks in {duration:.2f} seconds.")
    return True, duration

def search_vector_index(query_str: str, year_level_filter: str | None = None, k=3) -> list[str]:
    """Searches FAISS index. Returns text of top k chunks."""
    if not os.path.exists(VECTOR_INDEX_FILE) or not os.path.exists(VECTOR_MAPPING_FILE):
        print("VectorRetriever: Index or mapping file not found. Please create index first.")
        return []
    try:
        index = faiss.read_index(VECTOR_INDEX_FILE)
        with open(VECTOR_MAPPING_FILE, 'rb') as f:
            mapping = pickle.load(f) # This is a list of dicts
        
        model = SentenceTransformer(MODEL_NAME)
        query_vector = model.encode([query_str]).astype('float32')

        print(f"VectorRetriever: Searching for query: {query_str}")
        distances, indices = index.search(query_vector, k * 5) # Retrieve more to filter by year if needed

        retrieved_chunks = []
        if len(indices[0]) == 0 or indices[0][0] < 0:
            print(f"VectorRetriever: No results found for query: {query_str}")
            return []

        # Optional: Filter by year_level_filter if provided
        target_year_short_name = YEAR_FOLDER_MAP.get(year_level_filter) if year_level_filter else None

        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < 0 or idx >= len(mapping): continue # Invalid index

            hit_details = mapping[idx]
            if target_year_short_name and hit_details['year'] != target_year_short_name:
                # If year filter is active and this chunk doesn't match, skip it
                # This is a simple post-filter. More advanced would integrate into FAISS if possible or pre-filter mapping.
                continue 
            
            retrieved_chunks.append(hit_details['chunk_text'])
            print(f"VectorRetriever: Match from '{hit_details['path']}' (Distance: {distances[0][i]:.4f}, Year: {hit_details['year']})")
            if len(retrieved_chunks) >= k:
                break # Got enough results for the target year or in general
        
        return retrieved_chunks

    except Exception as e:
        print(f"VectorRetriever: Error during vector search: {e}")
        return []