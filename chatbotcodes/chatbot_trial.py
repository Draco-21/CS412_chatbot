# chatbot_trial.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import streamlit as st
import google.generativeai as genai
import time
import hashlib
from functools import lru_cache
import tenacity
from retrievalAlgorithms.resource_retriever import SmartResourceRetriever
import concurrent.futures
from typing import List, Dict, Optional

st.set_page_config(page_title="Coding Assistant Bot", layout="wide")

# Initialize resource retriever
@st.cache_resource
def initialize_resource_retriever():
    retriever = SmartResourceRetriever()
    return retriever

def preload_resources(retriever):
    """Preload all resources into cache."""
    st.info("🔄 Initializing and caching resources... This may take a moment.")
    progress_bar = st.progress(0)
    
    # Get all files to process
    files_to_process = []
    total_files = 0
    
    for year_folder in ["Year 1", "Year 2", "Year 3", "Year 4"]:
        base_path = os.path.join("resources", year_folder)
        if not os.path.exists(base_path):
            continue
            
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.lower().endswith(('.pdf', '.txt', '.cpp', '.h', '.java', '.py', '.md')):
                    files_to_process.append(os.path.join(root, file))
                    total_files += 1

    if not files_to_process:
        st.warning("⚠️ No resource files found to cache.")
        return

    # Process files in parallel
    processed_files = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {
            executor.submit(retriever._read_file_content, file_path): file_path 
            for file_path in files_to_process
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                # The content is automatically cached in _read_file_content
                _ = future.result()
                processed_files += 1
                progress_bar.progress(processed_files / total_files)
            except Exception as e:
                print(f"Error preprocessing {file_path}: {e}")

    st.success(f"✅ Successfully cached {processed_files} resource files!")
    time.sleep(1)  # Give users a moment to see the success message
    st.empty()  # Clear the messages

# Initialize the retriever at startup
if 'resource_retriever' not in st.session_state:
    st.session_state.resource_retriever = initialize_resource_retriever()
    # Preload resources only once when the app starts
    if 'resources_preloaded' not in st.session_state:
        preload_resources(st.session_state.resource_retriever)
        st.session_state.resources_preloaded = True

# Cache for API responses
if 'response_cache' not in st.session_state:
    st.session_state.response_cache = {}

# Rate limiting settings
if 'last_api_call' not in st.session_state:
    st.session_state.last_api_call = 0
MIN_TIME_BETWEEN_CALLS = 0.1  # 100ms minimum between API calls

def get_cache_key(prompt, year_level, local_context=None):
    """Generate a unique cache key for a prompt and year level."""
    context_str = str(local_context)[:100] if local_context else ""
    combined = f"{prompt}_{year_level}_{context_str}"
    return hashlib.md5(combined.encode()).hexdigest()

# --- API Key Management ---
def get_api_key():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        if 'GOOGLE_API_KEY' in st.session_state:
            api_key = st.session_state.GOOGLE_API_KEY
        else:
            st.sidebar.error("🔑 Google API Key not found!")
            api_key_input = st.sidebar.text_input("Enter Google API Key:", type="password", key="api_key_input_main")
            if api_key_input:
                st.session_state.GOOGLE_API_KEY = api_key_input
                os.environ["GOOGLE_API_KEY"] = api_key_input # Set for current session
                st.sidebar.success("✅ API Key set!")
                return api_key_input
            return None
    return api_key

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type((Exception)),
    before_sleep=lambda retry_state: print(f"Retrying API call in {retry_state.next_action.sleep} seconds...")
)
def make_api_call(model, final_prompt, safety_settings, gemini_history):
    """Make an API call with retry logic."""
    chat = model.start_chat(history=gemini_history)
    return chat.send_message(final_prompt, safety_settings=safety_settings)

# --- Gemini API Integration ---
def call_external_api(user_message: str, conversation_history: List[Dict], year_level: str, local_context: Optional[str] = None) -> str:
    """
    Calls the Google Gemini API to get a chatbot response with optimizations.
    
    Args:
        user_message (str): The user's question
        conversation_history (List[Dict]): Previous conversation messages
        year_level (str): The student's academic year level
        local_context (Optional[str]): Any relevant local resource context found
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

        # Get year-level specific instruction
        instruction_prefix = get_year_level_instruction(year_level)
        
        # Construct prompt with local context if available
        if local_context:
            final_prompt = f"""{instruction_prefix}

Local Resource Context:
{local_context}

Please use the above local resource information to help answer this question:
{user_message}

If the local resources don't fully answer the question, supplement with your knowledge but clearly indicate which parts come from course materials vs. your general knowledge."""
        else:
            final_prompt = f"{instruction_prefix}\n\nStudent's question: {user_message}"

        # Prepare history for Gemini API
        gemini_history = []
        for msg in conversation_history[-5:]:  # Only use last 5 messages for context
            role = 'user' if msg['role'] == 'user' else 'model'
            gemini_history.append({'role': role, 'parts': [msg['content']]})

        # Start chat with history and send the message
        chat = model.start_chat(history=gemini_history)
        
        # Stream the response
        response = chat.send_message(final_prompt, stream=True)
        
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

@lru_cache(maxsize=4)
def get_instruction_prefix(year_level):
    """Cache and return instruction prefixes for different year levels."""
    if year_level == "Year 1 Certificate":
        return """You are helping a Year 1 (Certificate) student. Keep explanations very simple.
                Use basic C++.
                Focus on:
                - Basic programming concepts (if-else, loops)
                - Simple data structures (arrays, linked lists)
                - Basic OOP concepts
                - Fundamental algorithms (sorting, searching)
                Student's question: """
    elif year_level == "Year 2 Diploma":
        return """You are helping a Year 2 (Diploma) student. Provide clear, practical examples in Java.
                Focus on:
                - Advanced data structures
                - Basic web development
                - Database fundamentals
                - Android development basics
                Student's question: """
    elif year_level == "Year 3 Degree":
        return """You are helping a Year 3 (Degree) student. Be detailed and discuss efficiency.
                Focus on:
                - Network programming
                - Distributed systems
                - Advanced frameworks
                - System design
                Student's question: """
    elif year_level == "Year 4 Postgraduate Diploma":
        return """You are helping a Year 4 (Postgraduate Diploma) student. Offer in-depth analysis.
                Focus on:
                - Machine learning algorithms
                - Advanced optimization
                - Neural networks
                - Research-level concepts
                Student's question: """
    else:
        return "You are a helpful coding assistant. Student's question: "

YEAR_LEVELS = ["Year 1 Certificate", "Year 2 Diploma", "Year 3 Degree", "Year 4 Postgraduate Diploma"]

if "selected_year" not in st.session_state:
    st.session_state.selected_year = YEAR_LEVELS[0]
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your coding assistant. I'll search through local resources first, then use my knowledge to help you. How can I help you today?"}]

def display_year_level_selector():
    st.sidebar.header("Student Year Level")
    selected_year = st.sidebar.selectbox("Select year:", options=YEAR_LEVELS, index=YEAR_LEVELS.index(st.session_state.selected_year))
    if selected_year != st.session_state.selected_year:
        st.session_state.selected_year = selected_year

# --- Main App Logic ---
api_key = get_api_key() # Get API key at the start

st.title("Coding Assistant Chatbot 🧪")
st.caption("Powered by Smart Resource Search and Gemini")

display_year_level_selector()
st.sidebar.write(f"Current Year: {st.session_state.selected_year}")

# Add cache status indicator in sidebar
if st.session_state.resources_preloaded:
    st.sidebar.success("📚 Resources cached and ready!")
    cached_files = len(st.session_state.resource_retriever._content_cache)
    st.sidebar.info(f"📂 {cached_files} files in cache")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and response
if prompt := st.chat_input("Ask your coding question..." if api_key else "Please enter API key in sidebar to chat"):
    if not api_key:
        st.warning("Please enter your Google API Key in the sidebar first.")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching local resources..."):
                current_year = st.session_state.selected_year
                
                # First search local resources
                matches = st.session_state.resource_retriever.search_resources(prompt, current_year)
                
                # Try to generate response from local resources first
                local_response = None
                if matches:
                    local_response = st.session_state.resource_retriever.generate_local_response(prompt, matches, current_year)
                
                if local_response:
                    st.success("✨ Response generated from your course materials!")
                    st.markdown(local_response)
                    bot_response = local_response
                else:
                    st.info("📡 No complete answer found in local resources. Consulting Gemini API...")
                    # Format local context for API if available
                    local_context = st.session_state.resource_retriever.format_results(matches) if matches else None
                    
                    # Get response from API with local context if found
                    with st.spinner("🧠 Building your answer..."):
                        api_history = [msg for msg in st.session_state.messages[:-1] if msg["role"] != "system"]
                        bot_response = call_external_api(prompt, api_history, current_year, local_context)
                        st.markdown(bot_response)

        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": bot_response})