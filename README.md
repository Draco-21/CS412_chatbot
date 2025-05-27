# Coding Assistant Chatbot

A smart coding assistant that helps students learn programming concepts based on their academic level. The chatbot provides language-specific and framework-specific guidance by searching through relevant educational resources.

## Features

- Year-level based response complexity (Year 1-4)
- Automatic language detection (C++, Java, Python)
- Framework detection (e.g., Spring Boot)
- Resource-based responses from curated educational materials
- Clean and simple user interface

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your Google API key:
   - Create a `.env` file in the project root
   - Add your Gemini API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

3. Organize your resources:
   - Place educational materials in the `resources` folder
   - Use the following structure:
     ```
     resources/
     ├── Year 1/  # C++ materials
     ├── Year 2/  # Java and Spring Boot materials
     ├── Year 3/  # Python materials
     └── Year 4/  # Advanced topics
     ```

## Usage

1. Start the application:
   ```bash
   streamlit run chatbot.py
   ```

2. Select your year level from the dropdown menu
   - Year 1: Simple explanations for beginners
   - Year 2: Moderate technical terms
   - Year 3: Technical terminology
   - Year 4: Advanced concepts

3. Ask your coding questions
   - For basic language questions (e.g., "show me a simple Java loop")
   - For framework-specific questions (e.g., "create a Spring Boot REST API")
   - The chatbot will automatically search the appropriate resources based on the language/framework detected

## Note

The response complexity is determined by the selected year level, while the content source is determined by the programming language or framework mentioned in your question. For example, if you're in Year 1 but ask about Java, it will give you Java content from Year 2 resources but explained in simpler terms.