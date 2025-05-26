# keyword_retriever.py
import os
import time # <--- Add this import
from whoosh.index import create_in, open_dir, EmptyIndexError
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.analysis import StemmingAnalyzer
from text_extractor import extract_text

RESOURCES_BASE_FOLDER = "resources"
KEYWORD_INDEX_DIR = "keyword_index"
YEAR_FOLDER_MAP = {
    "Year 1 Certificate": "Year 1",
    "Year 2 Diploma": "Year 2",
    "Year 3 Degree": "Year 3",
    "Year 4 Postgraduate Diploma": "Year 4"
}

def get_keyword_schema():
    stem_analyzer = StemmingAnalyzer()
    return Schema(
        path=ID(stored=True, unique=True),
        year=ID(stored=True),
        content=TEXT(stored=True, analyzer=stem_analyzer)
    )

def create_keyword_index() -> tuple[bool, float]: # <--- Modified return type
    """Builds or updates the Whoosh keyword index from the resources folder."""
    start_time = time.time() # <--- Record start time

    if not os.path.exists(RESOURCES_BASE_FOLDER):
        print(f"KeywordRetriever: Base resources folder '{RESOURCES_BASE_FOLDER}' not found.")
        return False, 0.0
    if not os.path.exists(KEYWORD_INDEX_DIR):
        os.mkdir(KEYWORD_INDEX_DIR)
        print(f"KeywordRetriever: Created index directory: '{KEYWORD_INDEX_DIR}'.")

    print("KeywordRetriever: Initializing Whoosh index...")
    ix = create_in(KEYWORD_INDEX_DIR, get_keyword_schema())
    # Using more RAM for writer, and multiple processors if available
    writer = ix.writer(limitmb=256, procs=os.cpu_count(), multisegment=True) 
    
    print("KeywordRetriever: Starting to index resources...")
    indexed_count = 0
    processed_files = 0

    for year_str, year_folder_short_name in YEAR_FOLDER_MAP.items():
        year_path = os.path.join(RESOURCES_BASE_FOLDER, year_folder_short_name)
        if os.path.isdir(year_path):
            print(f"KeywordRetriever: Processing folder: {year_path}")
            for root, _, files in os.walk(year_path):
                for filename in files:
                    processed_files += 1
                    file_path = os.path.join(root, filename)
                    content = extract_text(file_path)
                    if content:
                        try:
                            writer.add_document(
                                path=file_path,
                                year=year_folder_short_name,
                                content=content
                            )
                            indexed_count += 1
                        except Exception as e:
                            print(f"KeywordRetriever: Failed to add document {file_path} to index: {e}")
        else:
            print(f"KeywordRetriever: Year folder not found: {year_path}")

    print(f"KeywordRetriever: Committing {indexed_count} documents from {processed_files} files processed...")
    writer.commit()
    
    end_time = time.time() # <--- Record end time
    duration = end_time - start_time # <--- Calculate duration
    print(f"KeywordRetriever: Indexing complete. Indexed {indexed_count} documents in {duration:.2f} seconds.")
    return True, duration # <--- Return success and duration

# --- search_keyword_index function remains the same ---
def search_keyword_index(query_str: str, year_level_filter: str | None = None, result_limit=1) -> str | None:
    if not os.path.exists(KEYWORD_INDEX_DIR) or not os.path.exists(os.path.join(KEYWORD_INDEX_DIR, "_MAIN_LOCK")):
        print("KeywordRetriever: Index not found or appears empty. Please create it first.")
        return None
    try:
        ix = open_dir(KEYWORD_INDEX_DIR)
        with ix.searcher() as searcher:
            parser = MultifieldParser(["content", "path"], schema=ix.schema)
            query = parser.parse(query_str)
            print(f"KeywordRetriever: Searching for query: {query_str}")
            # Note: Year level filtering in Whoosh can be added here for more precision
            results = searcher.search(query, limit=result_limit)

            if not results:
                print(f"KeywordRetriever: No results found for query: {query_str}")
                return None
            best_hit = results[0]
            print(f"KeywordRetriever: Best match found in '{best_hit['path']}' (Score: {best_hit.score:.2f})")
            return best_hit['content']
    except EmptyIndexError:
        print("KeywordRetriever: Index is empty. Please create it first.")
        return None
    except Exception as e:
        print(f"KeywordRetriever: Error during Whoosh search: {e}")
        return None