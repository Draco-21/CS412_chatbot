import streamlit as st
import os
import google.generativeai as genai
from resource_retriever import find_answer_in_resources

# --- THIS SHOULD BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Coding Assistant Bot (Gemini)", layout="wide")

# --- Gemini API Integration ---
def call_external_api(user_message, conversation_history, year_level):
    """
    Calls the Google Gemini API to get a chatbot response.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Error: GOOGLE_API_KEY environment variable not set.")
        return "API key not configured. Please contact the administrator."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")

        # Detect language/framework from user message
        message_lower = user_message.lower()
        target_year = None
        
        # Language detection
        if "c++" in message_lower or "cpp" in message_lower:
            target_year = "Year 1"
        elif "java" in message_lower and not any(fw in message_lower for fw in ["springboot", "spring boot", "spring-boot"]):
            target_year = "Year 2"
        elif "python" in message_lower:
            target_year = "Year 3"
            
        # Framework detection overrides basic language detection
        if any(fw in message_lower for fw in ["springboot", "spring boot", "spring-boot"]):
            target_year = "Year 2"  # Spring Boot content is in Year 2

        # Search in resources first
        resource_content = find_answer_in_resources(user_message, target_year)

        # Prepare the instruction based on year level
        if year_level == "Year 1 Certificate":
            instruction_prefix = "Explain this in very simple terms, like explaining to a beginner. Use basic terminology and give step-by-step explanations: "
        elif year_level == "Year 2 Diploma":
            instruction_prefix = "Explain this in moderately technical terms, including some programming concepts: "
        elif year_level == "Year 3 Degree":
            instruction_prefix = "Explain this using proper technical terminology and programming concepts: "
        elif year_level == "Year 4 Postgraduate Diploma":
            instruction_prefix = "Provide an advanced explanation with in-depth technical details and best practices: "
        else:
            instruction_prefix = ""

        # Construct the final prompt
        if resource_content:
            final_prompt = f"{instruction_prefix}Based on this reference material:\n{resource_content}\n\nUser Question: {user_message}"
        else:
            final_prompt = f"{instruction_prefix}{user_message}"

        # Prepare conversation history
        gemini_history = [{'role': 'user' if msg['role'] == 'user' else 'model', 'parts': [msg['content']]} for msg in conversation_history]
        
        # Get response from Gemini
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(final_prompt)
        return response.text

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"I encountered an error. Please try again."

# --- Defining the year for the dropdown menu ---
YEAR_LEVELS = [
    "Year 1 Certificate",
    "Year 2 Diploma",
    "Year 3 Degree",
    "Year 4 Postgraduate Diploma"
]

# --- Initialize session state ---
if "selected_year" not in st.session_state:
    st.session_state.selected_year = YEAR_LEVELS[0]
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your coding questions today?"}]

# --- Function to display and manage year level selection ---
def display_year_level_selector():
    st.sidebar.selectbox(
        "Select your current year of study:",
        options=YEAR_LEVELS,
        key="selected_year"
    )

# --- Streamlit UI Code ---
st.title("Coding Assistant Chatbot ✨")
st.caption("Your AI Programming Assistant")

# Display the year level selector
display_year_level_selector()

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
        current_year = st.session_state.selected_year
        api_history = [msg for msg in st.session_state.messages[:-1]]
        bot_response = call_external_api(prompt, api_history, current_year)
        st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})