# chatbot_trial.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import streamlit as st
import google.generativeai as genai
from keyword_retriever import create_keyword_index, search_keyword_index
from vector_retriever import create_vector_index, search_vector_index
import time

st.set_page_config(page_title="Coding Assistant Bot (Gemini RAG)", layout="wide")

# --- Gemini API Integration (call_external_api - no changes needed) ---
def call_external_api(user_message, conversation_history, year_level):
    # (Make sure the logic for handling already augmented prompts vs direct prompts is correct)
    print(f"Calling Gemini API for: {user_message} with year level: {year_level}")
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Error: GOOGLE_API_KEY environment variable not set.")
        return "API key not configured. Please contact the administrator."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")

        instruction_prefix = ""
        if year_level == "Year 1 Certificate":
            instruction_prefix = """You are helping a Year 1 (Certificate) student. Keep explanations very simple.
                                    Use basic C++.
                                    if-else conditions, Loops such as While-Loop, For-Loop, and Do-While Loop.
                                    File input and output.
                                    Data Structures such as Arrays, Structs, Classes, Linked Lists, Heap, Stack, Queue, Hash Tables and Binary Trees.
                                    Pointers.
                                    Algorithms such as Sorting algorithms(Bubble Sort, Quick Sort, Big O Notation) and Searching algorithms (Linear Search, Binary Search and Recursive Binary Search).
                                    Recursions.
                                    Object Oriented Programming such as Inheritance, Polymorphism, Encapsulation, Abstraction, Constructors, Destructors, Friend Functions, Private Attributes, Public Attributes and Protected Attributes, Friend Functions, Overloading and Overriding.
                                    Standard Template Library (STL) such as Vectors, Maps, Sets, Iterators and Algorithms.
                                    Student's question: """
        elif year_level == "Year 2 Diploma":
            instruction_prefix = """You are helping a Year 2 (Diploma) student. Provide clear, practical examples. Try to use Java.
                                    Keep the instructions from Year 1 Certificate.
                                    Wombat Machine Learning, coding in binary numbers (0's and 1's)
                                    Assembly Language Programming.
                                    Stack and Queue, Push, Pop.
                                    Java with Basic Syntax, Dynamic Arrays, Priority Queue, Traveling Sales Person Problem, Deterministic and Non-Deterministic Algorithms, Empirical Testing, Sequential Search, Binary Search, Merge Sort, Quick Sort, Bubble Sort, Recursive vs Iterative Fibonacci, Greedy Algorithms.
                                    Insertion Sort, Multithreading.
                                    Android Studio Setup and Programming in Java and Kotlin.
                                    Ubuntu Setup and Wordpress programming.
                                    Java with Springboot framework, Maven and Gradle.
                                    ASP.net Coding with HTML, CSS and JavaScript.
                                    SQL with MySQL and SQLite.
                                    Student's question: """
        elif year_level == "Year 3 Degree":
            instruction_prefix = """You are helping a Year 3 (Degree) student. Be detailed, discuss concepts like efficiency if relevant. Answer in Any Programming Language Requested by User.
                                    Keep the instructions from Year 2 Diploma.
                                    Socket Programming in Python.
                                    Assigning Network Addresses, TCP/IP, UDP, Sockets, Ports, Client-Server Model.
                                    Topology Setup in CISCO Packet Tracer.
                                    gRPC implementation in Java with maven and gradle.
                                    Student's question: """
        elif year_level == "Year 4 Postgraduate Diploma":
            instruction_prefix = """You are helping a Year 4 (Postgraduate Diploma) student. Offer in-depth analysis, advanced concepts, and consider edge cases.
                                    Keep the instructions from Year 3 Degree.
                                    Perceptron Learning Algorithm.
                                    Neural Networks.
                                    Stochastic Gradient Descent.
                                    Logistic Regression.
                                    Linear Regression for Classification.
                                    Backward Propagation.
                                    Forward Propagation.
                                    Clustering and Optimization Algorithms.
                                    Student's question: """
        else:
            instruction_prefix = "You are a helpful coding assistant. Student's question: "
        # --- (End Instruction Prefix Logic) ---

        if "--- Start of Retrieved Content ---" in user_message or "Retrieved Context:" in user_message: # Check if it's an augmented prompt
            final_user_message_for_api = user_message
        else:
            final_user_message_for_api = f"{instruction_prefix}{user_message}"

        gemini_history = [{'role': 'user' if msg['role'] == 'user' else 'model', 'parts': [msg['content']]} for msg in conversation_history]
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
        
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(final_user_message_for_api, safety_settings=safety_settings)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        st.error(f"Sorry, I encountered an error trying to respond. Error: {e}")
        return "Apologies, I couldn't process your request due to an internal error."

# --- Defining the year for the dropdown menu ---
YEAR_LEVELS = ["Year 1 Certificate", "Year 2 Diploma", "Year 3 Degree", "Year 4 Postgraduate Diploma"]

# --- Initialize session state ---
if "selected_year" not in st.session_state:
    st.session_state.selected_year = YEAR_LEVELS[0]
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Select retrieval method, build index if needed, and ask!"}]

# Check for index existence more reliably
keyword_index_main_file = os.path.join("keyword_index", "_MAIN_LOCK") # Whoosh creates this
vector_index_file = "vector_index.faiss"
vector_mapping_file = "vector_mapping.pkl"

if "keyword_index_exists" not in st.session_state:
    st.session_state.keyword_index_exists = os.path.exists(keyword_index_main_file)
if "vector_index_exists" not in st.session_state:
    st.session_state.vector_index_exists = os.path.exists(vector_index_file) and os.path.exists(vector_mapping_file)


# --- Function to display and manage year level selection ---
def display_year_level_selector():
    st.sidebar.header("Student Year Level")
    selected_year = st.sidebar.selectbox("Select year:", options=YEAR_LEVELS, index=YEAR_LEVELS.index(st.session_state.selected_year))
    if selected_year != st.session_state.selected_year:
        st.session_state.selected_year = selected_year
        st.rerun()

# --- UI for Indexing and Method Selection ---
def display_retrieval_options():
    st.sidebar.header("Resource Retrieval")
    st.sidebar.write("Build Indexes (Run once or when files change):")

    if st.sidebar.button("Build Keyword Index"):
        with st.spinner("Building Keyword Index... This may take a while."):
            success, duration = create_keyword_index() # Capture duration
            if success:
                 st.session_state.keyword_index_exists = True
                 st.sidebar.success(f"Keyword Index Built! ({duration:.2f}s)")
            else:
                 st.sidebar.error("Keyword Index building failed. Check console.")
        time.sleep(2)
        # st.rerun() # Rerun only if necessary for UI update

    if st.sidebar.button("Build Vector Index"):
        # Add a warning for PyTorch related errors if they happen during model loading
        st.sidebar.warning("Vector indexing may take several minutes and requires internet for model download on first run. Ensure PyTorch/SentenceTransformers are correctly installed.")
        with st.spinner("Building Vector Index... This can be very time-consuming."):
            success, duration = create_vector_index() # Capture duration
            if success:
                 st.session_state.vector_index_exists = True
                 st.sidebar.success(f"Vector Index Built! ({duration:.2f}s)")
            else:
                 st.sidebar.error("Vector Index building failed. Check console.")
        time.sleep(2)
        # st.rerun()

    if st.session_state.keyword_index_exists:
        st.sidebar.caption("Keyword index appears built.")
    if st.session_state.vector_index_exists:
        st.sidebar.caption("Vector index appears built.")


    retrieval_method = st.sidebar.radio(
        "Select Retrieval Method:",
        ('Keyword Search RAG', 'Vector Search RAG', 'Gemini Only (No RAG)'),
        key='retrieval_method_selection' # Ensure a unique key
    )
    return retrieval_method

# --- Streamlit UI Code ---
st.title("Coding Assistant Chatbot 🧪")
st.caption("Powered by Streamlit, Gemini, and Advanced RAG")

display_year_level_selector()
selected_retrieval_method = display_retrieval_options()
st.sidebar.write(f"Current Year: {st.session_state.selected_year}")
st.sidebar.write(f"Using Method: {selected_retrieval_method}")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask your coding question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            current_year = st.session_state.selected_year
            api_history = [msg for msg in st.session_state.messages[:-1] if msg["role"] != "system"]
            retrieved_content = None
            bot_response = ""

            if selected_retrieval_method == 'Keyword Search RAG':
                if st.session_state.keyword_index_exists:
                    retrieved_content = search_keyword_index(prompt, current_year)
                else:
                    st.warning("Keyword Index not built. Falling back to Gemini Only.")
                    print("Keyword RAG selected, but index not found.")
            elif selected_retrieval_method == 'Vector Search RAG':
                if st.session_state.vector_index_exists:
                    retrieved_content = search_vector_index(prompt, current_year)
                else:
                    st.warning("Vector Index not built. Falling back to Gemini Only.")
                    print("Vector RAG selected, but index not found.")
            # If 'Gemini Only', retrieved_content remains None

            if retrieved_content:
                print(f"{selected_retrieval_method}: Local resource found, augmenting prompt for Gemini.")
                augmented_prompt = f"""You are a helpful assistant for a {current_year} student.
                Based *primarily* on the following retrieved context from local documents, answer the student's question.
                If the context is insufficient or doesn't directly answer, you may use your general knowledge but indicate that the provided context was limited.
                Do not mention the filename or the retrieval process itself (unless the document title is part of the context and relevant). Be conversational.

                Retrieved Context:
                ---
                {retrieved_content[:3500]} 
                ---
                Student's Question: "{prompt}"

                Answer:
                """
                bot_response = call_external_api(augmented_prompt, api_history, current_year)
            else:
                if selected_retrieval_method != 'Gemini Only (No RAG)':
                    print(f"{selected_retrieval_method}: No local resource found, or index missing. Calling API directly.")
                else:
                     print("Gemini Only selected. Calling API directly.")
                bot_response = call_external_api(prompt, api_history, current_year)

            st.markdown(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})