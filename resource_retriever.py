import os
from text_extractor import extract_text

RESOURCES_BASE_FOLDER = "resources"
YEAR_FOLDER_MAP = {
    "Year 1 Certificate": "Year 1",
    "Year 2 Diploma": "Year 2",
    "Year 3 Degree": "Year 3",
    "Year 4 Postgraduate Diploma": "Year 4"
}

def search_single_folder(folder_path: str, keywords: list[str]) -> str | None:
    """
    Searches recursively within a single folder path for files matching keywords.
    Returns the full content of the first matching file found.
    """
    if not os.path.isdir(folder_path):
        print(f"ResourceRetriever: Search path {folder_path} not found.")
        return None

    try:
        for root, _, files in os.walk(folder_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                content = extract_text(file_path)
                
                if content:
                    # Convert content to lowercase for case-insensitive search
                    content_lower = content.lower()
                    # Check if ALL keywords are present
                    if all(keyword.lower() in content_lower for keyword in keywords):
                        print(f"ResourceRetriever: Found match in '{file_path}'")
                        return content

    except Exception as e:
        print(f"ResourceRetriever: Error during search in {folder_path}: {e}")
        return None

    return None

def find_answer_in_resources(user_message: str, target_year: str | None = None) -> str | None:
    """
    Searches for relevant content in resource files based on the user's message and target year.
    """
    print(f"ResourceRetriever: Searching for '{user_message}' in year: {target_year}")

    # Extract keywords from user message
    keywords = [word.strip() for word in user_message.lower().split() if len(word.strip()) > 2]
    if not keywords:
        print("ResourceRetriever: No valid keywords in user message.")
        return None
    
    print(f"ResourceRetriever: Using keywords: {keywords}")

    # If target year is specified, search there first
    if target_year and target_year in YEAR_FOLDER_MAP.values():
        folder_path = os.path.join(RESOURCES_BASE_FOLDER, target_year)
        result = search_single_folder(folder_path, keywords)
        if result:
            return result

    # If no result found in target year or no target year specified,
    # search through all years in order
    for year_folder in YEAR_FOLDER_MAP.values():
        if year_folder != target_year:  # Skip if we already searched this year
            folder_path = os.path.join(RESOURCES_BASE_FOLDER, year_folder)
            result = search_single_folder(folder_path, keywords)
            if result:
                return result

    print(f"ResourceRetriever: No match found for '{user_message}'")
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

    # Example: Test a query that might exist in Year 2 but searched from Year 1
    # Ensure you have a 'Year 2' folder with a 'java_syntax.txt' file for this test
    # year2_path = os.path.join(RESOURCES_BASE_FOLDER, "Year 2")
    # if not os.path.exists(year2_path):
    #     os.makedirs(year2_path)
    # with open(os.path.join(year2_path, "java_syntax.txt"), "w") as f:
    #      f.write("Java syntax is similar to C++ but has key differences.")

    # print("\nTesting with 'Java syntax' from Year 1")
    # result2 = find_answer_in_resources("Java syntax", "Year 1 Certificate")
    # if result2:
    #      print("\n--- Search Result Found ---")
    #      print(result2[:500] + "...")
    # else:
    #      print("\nNo local result found.")