import streamlit as st
import os
import google.generativeai as genai
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Dict, List, Optional, Tuple
import re
from retrievalAlgorithms.keyword_retriever import search_keyword_index, create_keyword_index
from retrievalAlgorithms.text_extractor import extract_text

# --- THIS SHOULD BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Coding Assistant Bot (Gemini)", layout="wide")

# Cache configuration
CACHE_FILE = "response_cache.json"
CACHE_EXPIRY = 3600  # 1 hour in seconds
CODE_SNIPPET_CACHE_FILE = "code_snippet_cache.json"

def load_cache() -> Dict:
    """Load the response cache from file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
                current_time = time.time()
                return {k: v for k, v in cache.items() if current_time - v['timestamp'] < CACHE_EXPIRY}
        except Exception:
            return {}
    return {}

def save_cache(cache: Dict):
    """Save the response cache to file."""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"Error saving cache: {e}")

def get_cache_key(user_message: str, year_level: str) -> str:
    """Generate a cache key for the user message and year level."""
    return hashlib.md5(f"{user_message}:{year_level}".encode()).hexdigest()

def extract_code_snippets(text: str) -> List[Tuple[str, str]]:
    """Extract code snippets from text with their language."""
    code_pattern = r"```(\w+)?\n(.*?)\n```"
    matches = re.finditer(code_pattern, text, re.DOTALL)
    return [(match.group(1) or "python", match.group(2)) for match in matches]

def format_code_snippets(text: str) -> str:
    """Format code snippets with syntax highlighting."""
    code_pattern = r"```(\w+)?\n(.*?)\n```"
    
    def replace_code(match):
        lang = match.group(1) or "python"
        code = match.group(2)
        return f"```{lang}\n{code}\n```"
    
    return re.sub(code_pattern, replace_code, text, flags=re.DOTALL)

def get_relevant_resources(query: str, year_level: str) -> List[str]:
    """Get relevant resources using keyword search."""
    try:
        # Search in the keyword index
        results = search_keyword_index(query, year_level, result_limit=3)
        if results:
            return [results]
        return []
    except Exception as e:
        print(f"Error in keyword search: {e}")
        return []

def prepare_context(query: str, year_level: str) -> str:
    """Prepare context from relevant resources and year level."""
    resources = get_relevant_resources(query, year_level)
    context = ""
    
    if resources:
        context = "Relevant information from resources:\n" + "\n".join(resources)
    
    return context

# --- Gemini API Integration ---
def call_external_api(user_message: str, conversation_history: List[Dict], year_level: str) -> str:
    """
    Calls the Google Gemini API to get a chatbot response with optimizations.
    """
    print(f"Calling Gemini API for: {user_message} with year level: {year_level}")

    # Check cache first
    cache = load_cache()
    cache_key = get_cache_key(user_message, year_level)
    if cache_key in cache:
        print("Cache hit!")
        return cache[cache_key]['response']

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Error: GOOGLE_API_KEY environment variable not set. Please set it before running.")
        return "API key not configured. Please contact the administrator."

    try:
        genai.configure(api_key=api_key)

        # Get relevant context from keyword search
        context = prepare_context(user_message, year_level)

        # Optimized model configuration
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )

        # Construct year-level specific instruction with context
        instruction_prefix = get_year_level_instruction(year_level)
        final_user_message_for_api = f"{instruction_prefix}\n\nContext:\n{context}\n\nStudent's question: {user_message}"

        # Prepare history for Gemini API
        gemini_history = []
        for msg in conversation_history[-5:]:  # Only use last 5 messages for context
            role = 'user' if msg['role'] == 'user' else 'model'
            gemini_history.append({'role': role, 'parts': [msg['content']]})

        # Start chat with history and send the message
        chat = model.start_chat(history=gemini_history)
        
        # Stream the response
        response = chat.send_message(final_user_message_for_api, stream=True)
        
        # Collect and format the response
        full_response = ""
        response_container = st.empty()
        
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                # Update the response in real-time with formatted code snippets
                response_container.markdown(format_code_snippets(full_response))

        # Cache the response
        cache[cache_key] = {
            'response': full_response,
            'timestamp': time.time()
        }
        save_cache(cache)

        return full_response

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        st.error(f"Sorry, I encountered an error trying to respond. Please check the terminal logs. Error: {e}")
        return "Apologies, I couldn't process your request due to an internal error."

def get_year_level_instruction(year_level: str) -> str:
    """Get the instruction prefix based on year level."""
    instructions = {
        "Year 1 Certificate": """You are helping a Year 1 (Certificate) student. Keep explanations very simple.
            Use basic C++.
            Focus on: if-else conditions, Loops, File I/O, Basic Data Structures, Pointers, Basic Algorithms.
            Always provide code examples when explaining concepts.
            Student's question: """,
        "Year 2 Diploma": """You are helping a Year 2 (Diploma) student. Provide clear, practical examples.
            Focus on: Java, Assembly, Data Structures, Algorithms, Android Development, Web Development.
            Include relevant code snippets and practical examples.
            Student's question: """,
        "Year 3 Degree": """You are helping a Year 3 (Degree) student. Be detailed, discuss efficiency.
            Focus on: Network Programming, System Design, Advanced Algorithms, Distributed Systems.
            Provide detailed code examples with explanations.
            Student's question: """,
        "Year 4 Postgraduate Diploma": """You are helping a Year 4 (Postgraduate Diploma) student.
            Focus on: Machine Learning, Neural Networks, Advanced Algorithms, Research Methods.
            Include code examples and mathematical explanations where relevant.
            Student's question: """
    }
    return instructions.get(year_level, "You are a helpful coding assistant. Always provide code examples when explaining concepts. Student's question: ")

# --- Streamlit UI Code ---
YEAR_LEVELS = [
    "Year 1 Certificate",
    "Year 2 Diploma",
    "Year 3 Degree",
    "Year 4 Postgraduate Diploma"
]

if "selected_year" not in st.session_state:
    st.session_state.selected_year = YEAR_LEVELS[0]

def display_year_level_selector():
    """Displays the year level selectbox and manages its state."""
    st.sidebar.header("Student Year Level")
    selected_year_from_widget = st.sidebar.selectbox(
        "Select your current year of study:",
        options=YEAR_LEVELS,
        index=YEAR_LEVELS.index(st.session_state.selected_year)
    )

    if selected_year_from_widget != st.session_state.selected_year:
        st.session_state.selected_year = selected_year_from_widget
        st.rerun()

# Main UI
st.title("Coding Assistant Chatbot ✨")
st.caption("Powered by Streamlit and Gemini")

display_year_level_selector()
st.sidebar.write(f"Current level: {st.session_state.selected_year}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you with your coding questions today?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(format_code_snippets(message["content"]))

# Chat input
if prompt := st.chat_input("Ask your coding question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            api_history = [msg for msg in st.session_state.messages if msg["role"] != "system"]
            current_year = st.session_state.selected_year
            bot_response = call_external_api(prompt, api_history, current_year)
            st.markdown(format_code_snippets(bot_response))

    st.session_state.messages.append({"role": "assistant", "content": bot_response})