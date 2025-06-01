# keyword_retriever.py
import os
import time
from whoosh.index import create_in, open_dir, EmptyIndexError
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.analysis import StemmingAnalyzer
from retrievalAlgorithms.text_extractor import extract_text, chunk_text

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
        doc_path=ID(stored=True),
        year=ID(stored=True),
        chunk_id=ID(stored=True, unique=True),
        content=TEXT(stored=True, analyzer=stem_analyzer)
    )

def cleanup_index():
    """Clean up any incomplete index files."""
    try:
        if os.path.exists(KEYWORD_INDEX_DIR):
            temp_files = ["_MAIN_LOCK", "MAIN.tmp"]
            for temp_file in temp_files:
                file_path = os.path.join(KEYWORD_INDEX_DIR, temp_file)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Cleaned up {file_path}")
                    except Exception as e:
                        print(f"Warning: Could not remove {file_path}: {e}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def create_keyword_index() -> tuple[bool, float]:
    """Builds or updates the Whoosh keyword index from the resources folder."""
    try:
        start_time = time.time()
        cleanup_index()

        if not os.path.exists(RESOURCES_BASE_FOLDER):
            print(f"KeywordRetriever: Base resources folder '{RESOURCES_BASE_FOLDER}' not found.")
            return False, 0.0

        try:
            os.makedirs(KEYWORD_INDEX_DIR, exist_ok=True)
            test_file = os.path.join(KEYWORD_INDEX_DIR, "test_write")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            print(f"KeywordRetriever: Directory not writable: {e}")
            return False, 0.0

        print(f"KeywordRetriever: Using index directory: '{KEYWORD_INDEX_DIR}'.")
        print("KeywordRetriever: Initializing Whoosh index...")

        try:
            ix = create_in(KEYWORD_INDEX_DIR, get_keyword_schema())
            writer = ix.writer(limitmb=256, procs=os.cpu_count(), multisegment=True)
        except Exception as e:
            print(f"KeywordRetriever: Failed to create index: {e}")
            cleanup_index()
            return False, 0.0

        print("KeywordRetriever: Starting to index resources...")
        indexed_count = 0
        processed_files = 0

        try:
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
                                        doc_path=file_path,
                                        year=year_folder_short_name,
                                        chunk_id=f"{file_path}_{indexed_count}",
                                        content=content
                                    )
                                    indexed_count += 1
                                except Exception as e:
                                    print(f"KeywordRetriever: Failed to add document {file_path} to index: {e}")
                else:
                    print(f"KeywordRetriever: Year folder not found: {year_path}")

            print(f"KeywordRetriever: Committing {indexed_count} documents from {processed_files} files processed...")
            writer.commit()
            duration = time.time() - start_time
            print(f"KeywordRetriever: Indexing complete. Indexed {indexed_count} documents in {duration:.2f} seconds.")
            return True, duration

        except Exception as e:
            print(f"Error during indexing: {e}")
            writer.cancel()
            cleanup_index()
            return False, 0

    except KeyboardInterrupt:
        print("\nIndex creation interrupted. Cleaning up...")
        cleanup_index()
        return False, 0
    except Exception as e:
        print(f"Error creating index: {e}")
        cleanup_index()
        return False, 0

def search_keyword_index(query_str: str, year_level_filter: str | None = None, k=3) -> list[str]:
    """Searches the Whoosh index for chunks. Returns content of top k chunks."""
    if not os.path.exists(KEYWORD_INDEX_DIR) or not os.path.exists(os.path.join(KEYWORD_INDEX_DIR, "_MAIN_LOCK")):
        print("KeywordRetriever: Index not found. Please create it first.")
        return []
    try:
        ix = open_dir(KEYWORD_INDEX_DIR)
        with ix.searcher() as searcher:
            parser = MultifieldParser(["content"], schema=ix.schema, group=QueryParser.OrGroup)
            query = parser.parse(query_str)
            
            print(f"KeywordRetriever: Searching for query: {query_str}")
            results = searcher.search(query, limit=k)

            if not results:
                print(f"KeywordRetriever: No results found for query: {query_str}")
                return []
            
            retrieved_chunks = []
            for hit in results:
                retrieved_chunks.append(hit['content'])
                print(f"KeywordRetriever: Match from '{hit['doc_path']}' (Score: {hit.score:.2f})")
            return retrieved_chunks
    except EmptyIndexError:
        print("KeywordRetriever: Index is empty.")
        return []
    except Exception as e:
        print(f"KeywordRetriever: Error during Whoosh search: {e}")
        return []