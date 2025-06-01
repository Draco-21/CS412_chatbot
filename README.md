<<<<<<< HEAD
# Smart Educational Coding Assistant

A context-aware chatbot designed to assist students with programming queries across different academic years. The system intelligently searches through local educational resources before leveraging the Gemini API for comprehensive responses.

## 🌟 Features

### Smart Resource Retrieval
- **Context-Aware Search**: Understands the academic context (year level) and programming domain
- **Intelligent Path Selection**: Prioritizes relevant folders based on query context
- **Framework Recognition**: Automatically detects and handles framework-specific queries
- **Optimized Performance**: Uses caching and parallel processing for fast responses

### Year-Level Adaptation
- **Year 1 (Certificate)**: Focus on C++ basics and fundamental programming concepts
- **Year 2 (Diploma)**: Java, Spring Boot, Android development, and basic web technologies
- **Year 3 (Degree)**: Advanced web frameworks, network programming, and system design
- **Year 4 (Postgraduate)**: Machine learning, advanced algorithms, and research topics

### Technical Features
- **Resource Caching**: Pre-loads and caches resources for instant responses
- **Parallel Processing**: Multi-threaded file processing for improved performance
- **Smart Content Extraction**: Handles multiple file formats (PDF, code files, docs)
- **Relevance Scoring**: Weighted scoring system for accurate resource matching

## 📋 Prerequisites

```bash
# Python version
Python 3.8+

# Required packages
streamlit
google-generativeai
pdfplumber
python-docx
python-pptx
PyPDF2
tenacity
```

## 🚀 Quick Start

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd CS412_Chatbot-Project
   ```

2. **Install Dependencies**
=======
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
>>>>>>> 896b7a0c70ecf3b0f5ef62338d1c804066d17aee
   ```bash
   pip install -r requirements.txt
   ```

<<<<<<< HEAD
3. **Set Up Resources**
   - Place educational materials in the `resources` folder
   - Organize by year level:
     ```
     resources/
     ├── Year 1/
     │   ├── C++ Docs/
     │   ├── CS111 Lectures/
     │   └── Practical/
     ├── Year 2/
     │   ├── Java DSA/
     │   ├── Spring Boot/
     │   └── Android Studio/
     ├── Year 3/
     │   ├── Flask Framework/
     │   ├── CISCO/
     │   └── Socket Programming/
     └── Year 4/
         └── ML Resources/
     ```

4. **Set Up Environment Variables**
   ```bash
   # Set your Google API key
   export GOOGLE_API_KEY=your_api_key_here
   ```

5. **Run the Application**
   ```bash
   streamlit run chatbot_trial.py
   ```

## 💡 Usage

1. **Initial Setup**
   - Launch the application
   - Enter your Google API key in the sidebar
   - Wait for resource caching to complete

2. **Using the Chatbot**
   - Select the appropriate year level
   - Type your programming question
   - The system will:
     1. Search local resources first
     2. Use Gemini API if needed
     3. Provide a comprehensive response

3. **Resource Management**
   - Monitor cache status in the sidebar
   - View number of cached resources
   - Check resource retrieval performance

## 🔧 Project Structure

```
CS412_Chatbot-Project/
├── chatbot_trial.py        # Main application file
├── resource_retriever.py   # Smart resource retrieval system
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
└── resources/             # Educational resources directory
```

## 📚 Key Components

### chatbot_trial.py
- Main application interface
- Handles user interactions
- Manages API integration
- Controls resource caching

### resource_retriever.py
- Smart resource retrieval system
- Context-aware search implementation
- File content caching
- Parallel processing

## ⚙️ Configuration

### Year Level Settings
```python
YEAR_LEVELS = [
    "Year 1 Certificate",
    "Year 2 Diploma",
    "Year 3 Degree",
    "Year 4 Postgraduate Diploma"
]
```

### Resource Types
- PDF Documents
- Source Code Files (.cpp, .java, .py)
- Text Documents
- PowerPoint Presentations
- Word Documents

## 🔍 Search Algorithm

The system uses a multi-step search process:
1. **Keyword Extraction**: Analyzes query for key terms
2. **Path Determination**: Selects relevant resource paths
3. **Content Search**: Parallel search through cached content
4. **Relevance Scoring**: Weighted scoring based on multiple factors
5. **Context Integration**: Combines local resources with API knowledge

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for advanced language processing
- Streamlit for the web interface
- Various Python libraries for file processing
=======
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
>>>>>>> 896b7a0c70ecf3b0f5ef62338d1c804066d17aee
