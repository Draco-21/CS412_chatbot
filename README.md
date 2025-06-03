# IntelliSE 📱

An intelligent AI-powered coding assistant chatbot built with Streamlit and Python. IntelliSE helps users learn programming concepts, debug code, and understand web development across multiple programming languages and frameworks.

## 🌟 Features

- **Multi-Language Support**: Handles Python, JavaScript, Java, C++, PHP, and more
- **Dynamic Web Development Tutorials**: Creates interactive tutorials for various frameworks
  - Python: Flask, Django, FastAPI
  - JavaScript: Express.js, Koa.js
  - PHP: Laravel
- **Adaptive Learning**: Adjusts explanations based on user's experience level
- **Real-time Code Generation**: Provides working code examples with explanations
- **Interactive Chat Interface**: User-friendly dark-themed chat interface
- **Context-Aware Responses**: Maintains conversation context for better assistance

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/IntelliSE.git
cd IntelliSE
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file and add your configurations
cp .env.example .env
```

### Running the Application

1. Start the Streamlit server:
```bash
streamlit run chatbot_trial.py
```

2. Open your browser and navigate to:
```
http://localhost:8501
```

## 🎯 Usage

1. **Select Your Experience Level**:
   - Year 1: Basic, beginner-friendly terms
   - Year 2: More technical terminology
   - Year 3: Professional concepts
   - Year 4: Advanced academic terms

2. **Ask Questions**:
   - Programming concepts
   - Code debugging
   - Web development tutorials
   - Framework-specific queries

3. **Interactive Learning**:
   - Follow generated tutorials
   - Copy and modify code examples
   - Get real-time explanations

## 🛠️ Project Structure

```
IntelliSE/
├── chatbot_trial.py        # Main application file
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── cleaned_resources/      # Resource files
├── chatbotcodes/          # Core chatbot modules
│   ├── query_analyzer.py   # Query analysis
│   ├── rag_generator.py    # Response generation
│   └── resource_retriever.py # Resource management
├── retrievalAlgorithms/   # Search algorithms
└── static/                # Static assets
```

## 🔧 Configuration

The application can be configured through environment variables:

```env
KMP_DUPLICATE_LIB_OK=TRUE
RESOURCES_DIR=cleaned_resources
# Add other configuration variables
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Streamlit for the web framework
- Python community for various libraries
- Contributors and testers

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## 🔄 Updates

The project is actively maintained. Check the repository for latest updates and features.

---
Built with ❤️ using Python and Streamlit
