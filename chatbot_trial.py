# chatbot_trial.py
import os
import sys
import logging
from typing import List, Optional, Union
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chatbot.log')
    ]
)
logger = logging.getLogger(__name__)

# Add project root and chatbotcodes to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
chatbot_dir = os.path.join(project_root, "chatbotcodes")
sys.path.extend([project_root, chatbot_dir])

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Define constants
RESOURCES_DIR = os.path.join(project_root, "cleaned_resources")

import streamlit as st
import time
from dotenv import load_dotenv
from retrievalAlgorithms.resource_retriever import SmartResourceRetriever, ResourceMatch
from chatbotcodes.query_analyzer import QueryAnalysis, QueryAnalyzer
from chatbotcodes.rag_generator import LocalResponseGenerator, APIResponseGenerator, format_response_for_display

# Load environment variables
load_dotenv()

# Configure page settings at the start of the file
st.set_page_config(
    page_title="Coding Assistant Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="✨"
)

# Custom CSS for better formatting
st.markdown("""
<style>
    .response-container {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #6c757d;
    }
    
    .advanced-example {
        border-left-color: #dc3545;
    }
    
    .intermediate-example {
        border-left-color: #ffc107;
    }
    
    /* Tutorial-specific styles */
    h1 {
        color: #2c3e50;
        font-size: 2.2em;
        margin-bottom: 1em;
        padding-bottom: 0.5em;
        border-bottom: 2px solid #eee;
    }
    
    h2 {
        color: #34495e;
        font-size: 1.8em;
        margin-top: 1.5em;
        margin-bottom: 0.8em;
    }
    
    h3 {
        color: #455a64;
        font-size: 1.4em;
        margin-top: 1.2em;
        margin-bottom: 0.5em;
    }
    
    .tutorial-section {
        margin-bottom: 2em;
        padding: 1em;
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .code-block {
        background-color: #1e1e1e;
        border-radius: 5px;
        padding: 12px;
        margin: 10px 0;
        color: #d4d4d4;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.9em;
        overflow-x: auto;
    }
    
    .code-block pre {
        margin: 0;
        padding: 0;
    }
    
    .code-explanation {
        background-color: #f8f9fa;
        border-left: 4px solid #17a2b8;
        padding: 1em;
        margin: 1em 0;
        border-radius: 0 4px 4px 0;
    }
    
    .setup-step {
        background-color: #f8f9fa;
        padding: 1em;
        margin: 0.5em 0;
        border-radius: 4px;
        border-left: 4px solid #28a745;
    }
    
    .setup-step code {
        background-color: #e9ecef;
        padding: 0.2em 0.4em;
        border-radius: 3px;
        font-family: 'Consolas', 'Monaco', monospace;
    }
    
    .component-requirements {
        background-color: #e8f4f8;
        padding: 0.8em;
        margin: 0.5em 0;
        border-radius: 4px;
    }
    
    .next-steps {
        background-color: #f8f9fa;
        padding: 1em;
        margin-top: 1em;
        border-radius: 4px;
        border-left: 4px solid #ffc107;
    }
    
    .next-steps ul {
        margin: 0;
        padding-left: 1.2em;
    }
    
    .language-label {
        display: inline-block;
        padding: 0.2em 0.6em;
        margin: 0 0.2em;
        border-radius: 3px;
        font-size: 0.8em;
        font-weight: bold;
        color: white;
    }
    
    .language-python { background-color: #3572A5; }
    .language-html { background-color: #e34c26; }
    .language-css { background-color: #563d7c; }
    .language-javascript { background-color: #f1e05a; color: black; }
    .language-bash { background-color: #333; }
    
    .file-structure {
        font-family: 'Consolas', 'Monaco', monospace;
        padding: 1em;
        background-color: #2c3e50;
        color: #fff;
        border-radius: 4px;
        margin: 1em 0;
    }
    
    .implementation-section {
        margin: 2em 0;
    }
    
    .implementation-section .title {
        display: flex;
        align-items: center;
        gap: 0.5em;
        margin-bottom: 1em;
    }
    
    .implementation-section .description {
        color: #666;
        margin-bottom: 1em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components with absolute path
@st.cache_resource
def initialize_components():
    return {
        'retriever': SmartResourceRetriever(resources_dir=RESOURCES_DIR),
        'analyzer': QueryAnalyzer(),
        'local_generator': LocalResponseGenerator(),
        'api_generator': APIResponseGenerator()
    }

# Get or initialize components
if 'components' not in st.session_state:
    st.session_state.components = initialize_components()

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = {
        'last_code': None,  # Last code example shown
        'last_topic': None,  # Last topic discussed
        'last_language': None,  # Last programming language used
        'last_explanation': None,  # Last explanation given
        'chat_history': [],  # List of all messages
        'context_stack': []  # Stack of conversation contexts
    }

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Year level selector
def display_year_level_selector():
    st.sidebar.markdown("""
        ### Year Level Selection
        Choose your year level to get explanations with appropriate terminology.
        - Year 1: Basic, beginner-friendly terms
        - Year 2: More technical terminology
        - Year 3: Professional concepts
        - Year 4: Advanced academic terms
        
        Note: You can ask about any programming language regardless of your year level!
    """)
    
    return st.sidebar.selectbox(
        "Select Your Year Level:",
        [
            "Year 1 Certificate",
            "Year 2 Diploma",
            "Year 3 Degree",
            "Year 4 Postgraduate Diploma"
        ],
        key="year_level"
    )

def format_resource_match(match: ResourceMatch, is_best_match: bool = False) -> str:
    """Format a resource match into a nice HTML display."""
    # For simple program requests, show a more concise response
    if 'simple' in match.title.lower() or 'basic' in match.title.lower():
        return f"""
        <div class="response-container">
            <div class="code-block">
                <pre><code class="{match.language}">{match.code}</code></pre>
            </div>
            <div style="margin-top: 10px;">
                <em>{match.purpose}</em>
            </div>
        </div>
        """
    
    # For other requests, show more comprehensive details
    topics_html = ' '.join([f'<span class="topic-tag">{topic}</span>' for topic in match.topics])
    
    # Add complexity indicators
    complexity_class = "advanced-example" if len(match.code.split('\n')) > 20 else "intermediate-example"
    
    return f"""
    <div class="response-container {complexity_class}">
        <div class="example-header">
            <span class="title">{match.title}</span>
            <div class="badges">
                <span class="language-label">{match.language}</span>
                {f'<span class="framework-label">{match.framework_name}</span>' if match.framework_name else ''}
            </div>
        </div>
        <div class="code-block">
            <pre><code class="{match.language}">{match.code}</code></pre>
        </div>
        <div style="margin-top: 10px;">
            <em>{match.purpose}</em>
            {f'<br><strong>Topics:</strong> {topics_html}' if topics_html else ''}
            {f'<br><strong>Context:</strong> {match.context}' if match.context else ''}
        </div>
        {f'<div class="implementation-notes">{match.notes}</div>' if hasattr(match, 'notes') and match.notes else ''}
    </div>
    """

def format_matches_for_display(matches: List[ResourceMatch]) -> str:
    """Format all matches into a nice display."""
    if not matches:
        return ""
    
    # Sort matches by score in descending order
    matches = sorted(matches, key=lambda x: x.score, reverse=True)
    
    # For simple program requests, just show the best match
    if any('simple' in match.title.lower() or 'basic' in match.title.lower() for match in matches):
        return format_resource_match(matches[0], is_best_match=True)
    
    # For other requests, show up to 3 matches based on relevance and complexity
    result = []
    result.append(format_resource_match(matches[0], is_best_match=True))
    
    # Show additional examples if they have a high score and add value
    for match in matches[1:3]:
        if match.score > 0.6 and len(match.code) > len(matches[0].code) * 0.5:  # Only show if significantly different
            result.append(format_resource_match(match, is_best_match=False))
    
    return "\n".join(result)

def get_technical_level(year_level: str) -> dict:
    """Return appropriate technical depth and terminology based on year level."""
    levels = {
        "Year 1 Certificate": {
            "depth": "basic",
            "terms": "beginner-friendly",
            "detail_level": "fundamental",
            "examples": "simple",
            "theory": "minimal"
        },
        "Year 2 Diploma": {
            "depth": "intermediate",
            "terms": "technical",
            "detail_level": "detailed",
            "examples": "practical",
            "theory": "moderate"
        },
        "Year 3 Degree": {
            "depth": "advanced",
            "terms": "professional",
            "detail_level": "comprehensive",
            "examples": "complex",
            "theory": "detailed"
        },
        "Year 4 Postgraduate Diploma": {
            "depth": "expert",
            "terms": "academic",
            "detail_level": "in-depth",
            "examples": "sophisticated",
            "theory": "extensive"
        }
    }
    return levels.get(year_level, levels["Year 1 Certificate"])

def format_code_explanation(code_parts: dict, tech_level: dict) -> str:
    """Format code explanations based on technical level."""
    explanations = []
    for component, details in code_parts.items():
        if tech_level["depth"] in ["basic", "intermediate"]:
            explanation = details["basic"]
        elif tech_level["depth"] == "advanced":
            explanation = details.get("advanced", details["basic"])
        else:
            explanation = details.get("expert", details.get("advanced", details["basic"]))
        
        explanations.append(f"\n{component}\n{explanation}")
    
    return "\n".join(explanations)

def get_default_language(year_level: str) -> str:
    """Get default programming language based on year level if none specified."""
    defaults = {
        "Year 1 Certificate": "c++",
        "Year 2 Diploma": "java",
        "Year 3 Degree": "python",
        "Year 4 Postgraduate Diploma": "python"
    }
    return defaults.get(year_level, "c++")

def detect_language_from_query(query: str, year_level: str) -> str:
    """Detect programming language from query, fallback to year-level default if none found."""
    query_lower = query.lower()
    
    # First, check for explicit language mentions
    language_keywords = {
        "c++": ["c++", "cpp"],
        "python": ["python", "py"],
        "javascript": ["javascript", "js", "node"],
        "java": ["java"],
        "c#": ["c#", "csharp", ".net"],
        "ruby": ["ruby", "rails"],
        "php": ["php"],
        "swift": ["swift", "ios"],
        "kotlin": ["kotlin", "android"],
        "go": ["golang", "go "],
        "rust": ["rust"],
        "matlab": ["matlab", "mat"],
        "html": ["html", "webpage", "web page"],
        "css": ["css", "stylesheet"],
        "dart": ["dart", "flutter"],
        "r": [" r ", "rlang"]
    }
    
    # Check for explicit framework mentions
    framework_to_language = {
        "django": "python",
        "flask": "python",
        "spring": "java",
        "react": "javascript",
        "angular": "javascript",
        "vue": "javascript",
        "express": "javascript",
        "rails": "ruby",
        "laravel": "php",
        "asp.net": "c#",
        "flutter": "dart"
    }
    
    # First check for explicit language mentions
    for lang, keywords in language_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return lang
            
    # Then check for framework mentions only if frameworks are explicitly asked about
    if "framework" in query_lower:
        for framework, lang in framework_to_language.items():
            if framework in query_lower:
                return lang
    
    # If no language is detected, return the default for the year level
    return get_default_language(year_level)

def search_resources_across_years(query: str, current_year_level: str) -> List[ResourceMatch]:
    """Search for resources across all year levels, prioritizing the current year level."""
    all_matches = []
    year_levels = ["Year 1 Certificate", "Year 2 Diploma", "Year 3 Degree", "Year 4 Postgraduate Diploma"]
    
    # First search in current year level
    matches = st.session_state.components['retriever'].search_resources(
        query=query,
        year_level=current_year_level
    )
    if matches:
        all_matches.extend(matches)
    
    # If no matches found in current year level, search in other year levels
    if not all_matches:
        for year in year_levels:
            if year != current_year_level:
                matches = st.session_state.components['retriever'].search_resources(
                    query=query,
                    year_level=year
                )
                if matches:
                    all_matches.extend(matches)
    
    return all_matches

def get_language_config():
    """Return configuration for all supported languages with their characteristics."""
    return {
        "python": {
            "file_extension": ".py",
            "comment_symbol": "#",
            "keywords": ["python", "py"],
            "frameworks": ["django", "flask", "fastapi", "pytorch", "tensorflow"],
            "hello_world": {
                "basic": """# Simple Python program
print("Hello World!")""",
                "intermediate": """def greet(message):
    print(message)

if __name__ == "__main__":
    greet("Hello World!")""",
                "advanced": """class Greeter:
    def __init__(self, message):
        self.message = message
    
    def greet(self):
        print(self.message)

if __name__ == "__main__":
    greeter = Greeter("Hello World!")
    greeter.greet()"""
            },
            "concepts": {
                "basic": ["print", "variables", "input"],
                "intermediate": ["functions", "classes", "modules"],
                "advanced": ["decorators", "generators", "metaclasses"]
            }
        },
        "java": {
            "file_extension": ".java",
            "comment_symbol": "//",
            "keywords": ["java", "jvm"],
            "frameworks": ["spring", "hibernate", "jakarta"],
            "hello_world": {
                "basic": """public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello World!");
    }
}""",
                "intermediate": """public class HelloWorld {
    private static String message = "Hello World!";
    
    public static void main(String[] args) {
        System.out.println(message);
    }
}""",
                "advanced": """public class HelloWorld {
    private interface Greeter {
        void greet();
    }
    
    private static class WorldGreeter implements Greeter {
        private final String message;
        
        public WorldGreeter(String message) {
            this.message = message;
        }
        
        @Override
        public void greet() {
            System.out.println(message);
        }
    }
    
    public static void main(String[] args) {
        Greeter greeter = new WorldGreeter("Hello World!");
        greeter.greet();
    }
}"""
            },
            "concepts": {
                "basic": ["classes", "methods", "printing"],
                "intermediate": ["inheritance", "interfaces", "collections"],
                "advanced": ["generics", "threads", "reflection"]
            }
        },
        "c++": {
            "file_extension": ".cpp",
            "comment_symbol": "//",
            "keywords": ["c++", "cpp"],
            "frameworks": ["qt", "boost", "opencv"],
            "hello_world": {
                "basic": """#include <iostream>

int main() {
    std::cout << "Hello World!" << std::endl;
    return 0;
}""",
                "intermediate": """#include <iostream>
#include <string>

int main() {
    std::string message = "Hello World!";
    std::cout << message << std::endl;
    return EXIT_SUCCESS;
}""",
                "advanced": """#include <iostream>
#include <memory>

class Greeter {
public:
    virtual ~Greeter() = default;
    virtual void greet() const = 0;
};

class HelloWorld : public Greeter {
public:
    void greet() const override {
        std::cout << "Hello World!" << std::endl;
    }
};

int main() {
    auto greeter = std::make_unique<HelloWorld>();
    greeter->greet();
    return EXIT_SUCCESS;
}"""
            },
            "concepts": {
                "basic": ["iostream", "main function", "basic output"],
                "intermediate": ["classes", "pointers", "references"],
                "advanced": ["templates", "STL", "smart pointers"]
            }
        },
        "javascript": {
            "file_extension": ".js",
            "comment_symbol": "//",
            "keywords": ["javascript", "js", "node", "nodejs"],
            "frameworks": ["react", "vue", "angular", "express"],
            "hello_world": {
                "basic": """console.log("Hello World!");""",
                "intermediate": """function greet(message) {
    console.log(message);
}

greet("Hello World!");""",
                "advanced": """class Greeter {
    constructor(message) {
        this.message = message;
    }
    
    greet() {
        console.log(this.message);
    }
}

const greeter = new Greeter("Hello World!");
greeter.greet();"""
            },
            "concepts": {
                "basic": ["console.log", "variables", "functions"],
                "intermediate": ["objects", "promises", "modules"],
                "advanced": ["async/await", "closures", "prototypes"]
            }
        },
        "html": {
            "file_extension": ".html",
            "comment_symbol": "<!--",
            "keywords": ["html", "webpage", "web page"],
            "frameworks": ["bootstrap", "tailwind", "material"],
            "hello_world": {
                "basic": """<!DOCTYPE html>
<html>
<head>
    <title>Hello World</title>
</head>
<body>
    <h1>Hello World!</h1>
</body>
</html>""",
                "intermediate": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hello World</title>
    <style>
        body { font-family: Arial, sans-serif; }
        h1 { color: blue; }
    </style>
</head>
<body>
    <h1>Hello World!</h1>
    <p>Welcome to my webpage!</p>
</body>
</html>""",
                "advanced": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hello World</title>
    <link rel="stylesheet" href="styles.css">
    <script defer src="app.js"></script>
</head>
<body>
    <header>
        <nav>
            <ul>
                <li><a href="#home">Home</a></li>
                <li><a href="#about">About</a></li>
            </ul>
        </nav>
    </header>
    <main>
        <h1>Hello World!</h1>
        <p>Welcome to my webpage!</p>
    </main>
    <footer>
        <p>&copy; 2024</p>
    </footer>
</body>
</html>"""
            },
            "concepts": {
                "basic": ["tags", "elements", "attributes"],
                "intermediate": ["semantic elements", "forms", "css"],
                "advanced": ["accessibility", "SEO", "responsive design"]
            }
        },
        "matlab": {
            "file_extension": ".m",
            "comment_symbol": "%",
            "keywords": ["matlab", "mat"],
            "frameworks": ["simulink", "app designer"],
            "hello_world": {
                "basic": """% Simple MATLAB program
disp('Hello World!')""",
                "intermediate": """function greet()
    message = 'Hello World!';
    disp(message)
end""",
                "advanced": """classdef Greeter
    properties
        message
    end
    
    methods
        function obj = Greeter(msg)
            obj.message = msg;
        end
        
        function greet(obj)
            disp(obj.message)
        end
    end
end"""
            },
            "concepts": {
                "basic": ["matrices", "basic operations", "plotting"],
                "intermediate": ["functions", "scripts", "data analysis"],
                "advanced": ["object-oriented", "GUI development", "toolboxes"]
            }
        },
        "dart": {
            "file_extension": ".dart",
            "comment_symbol": "//",
            "keywords": ["dart", "flutter"],
            "frameworks": ["flutter", "aqueduct"],
            "hello_world": {
                "basic": """void main() {
  print('Hello World!');
}""",
                "intermediate": """class Greeter {
  String message;
  
  Greeter(this.message);
  
  void greet() {
    print(message);
  }
}

void main() {
  var greeter = Greeter('Hello World!');
  greeter.greet();
}""",
                "advanced": """import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text('Hello World App'),
        ),
        body: Center(
          child: Text('Hello World!'),
        ),
      ),
    );
  }
}"""
            },
            "concepts": {
                "basic": ["variables", "functions", "control flow"],
                "intermediate": ["classes", "async", "libraries"],
                "advanced": ["widgets", "state management", "platform channels"]
            }
        }
    }

def get_language_info(language: str) -> dict:
    """Get language configuration for a specific language."""
    config = get_language_config()
    return config.get(language.lower(), {
        "file_extension": ".txt",
        "comment_symbol": "#",
        "keywords": [language.lower()],
        "frameworks": [],
        "hello_world": {
            "basic": f'print("Hello World!")',
            "intermediate": f'function greet() {{ print("Hello World!"); }}',
            "advanced": f'class Greeter {{ constructor() {{ print("Hello World!"); }} }}'
        },
        "concepts": {
            "basic": ["basic syntax", "output", "variables"],
            "intermediate": ["functions", "data structures", "modules"],
            "advanced": ["advanced features", "best practices", "optimization"]
        }
    })

def generate_simple_program_response(query: str, year_level: str) -> str:
    """Generate response for simple program requests."""
    tech_level = get_technical_level(year_level)
    language = detect_language_from_query(query, year_level)
    
    # Get language configuration
    lang_info = get_language_info(language)
    
    # Get appropriate example based on technical level
    if tech_level["depth"] == "basic":
        code = lang_info["hello_world"]["basic"]
    elif tech_level["depth"] == "intermediate":
        code = lang_info["hello_world"]["intermediate"]
    else:
        code = lang_info["hello_world"]["advanced"]
    
    # Get concepts for this level
    concepts = lang_info["concepts"][tech_level["depth"]]
    
    code_parts = {
        "Program Structure": {
            "basic": f"This is a basic {language} program showing core concepts: {', '.join(concepts)}",
            "advanced": f"This example demonstrates {language} concepts including: {', '.join(concepts)}",
            "expert": f"This implementation showcases advanced {language} features: {', '.join(concepts)}"
        }
    }
    
    return format_dynamic_response(code, code_parts, tech_level, language)

def format_dynamic_response(code: str, explanations: dict, tech_level: dict, language: str) -> str:
    """Format the final response with appropriate technical depth and user-friendly style."""
    
    # Define syntax highlighting based on language
    syntax_highlight = {
        "keyword": lambda x: f"`{x}`",
        "code_block": lambda x: f"```{language}\n{x}\n```",
        "inline_code": lambda x: f"`{x}`"
    }

    # Start with an engaging introduction based on technical level and language
    intros = {
        "basic": f"Let's create a simple program that will help you understand the basics of {language}. We'll break it down step by step so it's easy to follow.",
        "intermediate": f"Let's explore a {language} program that demonstrates some key programming concepts. I'll explain each part and introduce some best practices.",
        "advanced": f"Let's create a {language} program that showcases professional coding practices. We'll examine the structure and advanced concepts involved.",
        "expert": f"Let's implement a {language} program using industry best practices and design patterns. We'll analyze each component in detail."
    }
    
    response_parts = [intros[tech_level["depth"]]]
    
    # Add the initial code structure section
    response_parts.append("\nStep 1: Understanding the Structure")
    response_parts.append(f"\nA basic {language} program looks like this:")
    response_parts.append(syntax_highlight["code_block"](code))
    
    # Break down each component with appropriate technical depth
    response_parts.append("\nLet's break it down:")

    # Language-specific explanations for basic concepts
    if tech_level["depth"] == "basic":
        if language == "python":
            response_parts.extend([
                f"\n• {syntax_highlight['keyword']('print()')} : This is how we display text on the screen in Python.",
                f"\n• The text inside the quotes {syntax_highlight['keyword']('\"Hello World!\"')} is called a string - it's the message we want to show.",
                f"\n• Python is simple and doesn't need any special structure - we can write our code directly!"
            ])
        elif language == "java":
            response_parts.extend([
                f"\n• {syntax_highlight['keyword']('public class HelloWorld')} : In Java, we need to put our code inside a class.",
                f"\n• {syntax_highlight['keyword']('public static void main(String[] args)')} : This special method is where Java starts running our program.",
                f"\n• {syntax_highlight['keyword']('System.out.println()')} : This is how we print text to the screen in Java."
            ])
        elif language == "c++":
            response_parts.extend([
                f"\n• {syntax_highlight['keyword']('#include <iostream>')} : This line is like saying \"I need some tools from the iostream library.\"",
                f"\n• {syntax_highlight['keyword']('int main() { ... }')} : This is where your program starts and runs.",
                f"\n• {syntax_highlight['keyword']('return 0;')} : This tells the computer that your program finished successfully."
            ])
        elif language == "javascript":
            response_parts.extend([
                f"\n• {syntax_highlight['keyword']('console.log()')} : This is how we print text in JavaScript.",
                f"\n• The text in quotes is the message we want to display.",
                f"\n• JavaScript can run directly in a browser or using Node.js!"
            ])
        else:
            # Generic explanations for other languages
            response_parts.extend([
                f"\n• This is a basic {language} program that displays text.",
                f"\n• Most languages have a simple way to show output on the screen.",
                f"\n• The exact syntax varies by language, but the concept is the same."
            ])

    # Add language-specific tips
    response_parts.append("\nImportant Tips:")
    
    if language == "python":
        tips = [
            "- Use proper indentation (spaces) to structure your code",
            "- Add comments using #",
            "- Python doesn't need semicolons at the end of lines",
            "- Keep your code clean and readable"
        ]
    elif language == "java":
        tips = [
            "- Always end statements with a semicolon (;)",
            "- Class names should start with a capital letter",
            "- Save your file with the same name as your public class",
            "- Add comments using // or /* */"
        ]
    elif language == "c++":
        tips = [
            "- End statements with a semicolon (;)",
            "- Include necessary header files",
            "- Add comments using // or /* */",
            "- Don't forget to return from main()"
        ]
    elif language == "javascript":
        tips = [
            "- End statements with a semicolon (;) (optional but recommended)",
            "- Add comments using // or /* */",
            "- Use 'const' and 'let' instead of 'var'",
            "- Consider using strict mode with 'use strict'"
        ]
    else:
        # Generic tips that apply to most languages
        tips = [
            "- Keep your code organized and well-structured",
            "- Add comments to explain your code",
            "- Use consistent naming conventions",
            "- Test your program after writing it"
        ]
    
    response_parts.extend(tips)

    # Add an engaging conclusion based on level and language
    conclusions = {
        "basic": {
            "python": "\nWould you like to try adding user input with input() or learn about variables?",
            "java": "\nShall we try adding variables or creating a more interactive program?",
            "c++": "\nWould you like to learn about variables or user input with cin?",
            "javascript": "\nShall we try working with variables or handling user interactions?",
            "default": "\nWould you like to learn more about variables or user input?"
        },
        "intermediate": {
            "python": "\nShall we explore functions, lists, or working with files?",
            "java": "\nWould you like to learn about methods, arrays, or object-oriented concepts?",
            "c++": "\nShall we explore functions, arrays, or object-oriented programming?",
            "javascript": "\nWould you like to learn about functions, arrays, or DOM manipulation?",
            "default": "\nShall we explore more features like functions and data structures?"
        },
        "advanced": {
            "python": "\nWould you like to explore decorators, generators, or advanced OOP concepts?",
            "java": "\nShall we dive into interfaces, generics, or multithreading?",
            "c++": "\nWould you like to explore templates, STL, or memory management?",
            "javascript": "\nShall we explore promises, async/await, or advanced ES6+ features?",
            "default": "\nWould you like to explore advanced features of the language?"
        }
    }
    
    conclusion = conclusions[tech_level["depth"]].get(language, conclusions[tech_level["depth"]]["default"])
    response_parts.append(conclusion)

    return "\n".join(response_parts)

def detect_generic_intent(query: str) -> dict:
    """Detect if the query is a generic interaction like greetings or thanks."""
    query_lower = query.lower().strip()
    
    # Greeting patterns
    greetings = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 
        'good evening', 'howdy', 'what\'s up', 'sup'
    ]
    
    # Gratitude patterns
    thanks = [
        'thank', 'thanks', 'appreciate', 'grateful', 'thx',
        'ty', 'thank you'
    ]
    
    # Generic follow-up patterns
    follow_ups = [
        'can you explain', 'tell me more', 'elaborate', 'what about',
        'how about', 'is there', 'could you', 'please explain',
        'what do you mean', 'i don\'t understand', 'clarify'
    ]
    
    # Check each category
    if any(query_lower.startswith(g) for g in greetings):
        return {'type': 'greeting', 'confidence': 0.9}
    elif any(t in query_lower for t in thanks):
        return {'type': 'gratitude', 'confidence': 0.9}
    elif any(query_lower.startswith(f) for f in follow_ups):
        return {'type': 'follow_up', 'confidence': 0.8}
    
    return {'type': None, 'confidence': 0.0}

def generate_generic_response(intent: dict) -> str:
    """Generate appropriate responses for generic interactions."""
    import random
    
    responses = {
        'greeting': [
            "Hello! How can I help you with your programming questions today?",
            "Hi there! What programming challenge can I help you with?",
            "Hey! Ready to help you with your coding questions.",
            "Welcome! What would you like to learn about?"
        ],
        'gratitude': [
            "You're welcome! Let me know if you need anything else.",
            "Happy to help! Feel free to ask more questions.",
            "Glad I could help! What else would you like to know?",
            "No problem! Is there anything else you'd like to explore?"
        ]
    }
    
    if intent['type'] in responses:
        return random.choice(responses[intent['type']])
    return None

def handle_follow_up_question(query: str, year_level: str) -> Optional[str]:
    """Handle follow-up questions about previous responses."""
    query_lower = query.lower()
    context = st.session_state.conversation_context
    
    # Common follow-up patterns
    simplification_patterns = ['simpler', 'easier', 'basic', 'simple way', 'easier way']
    elaboration_patterns = ['more detail', 'explain more', 'tell me more', 'elaborate']
    alternative_patterns = ['another way', 'different way', 'other way', 'alternative']
    example_patterns = ['example', 'show me', 'demonstrate']
    
    # Check if we have context to work with
    if not context['context_stack']:
        return None
    
    current_context = context['context_stack'][-1]
    
    # Handle different types of follow-ups
    if any(pattern in query_lower for pattern in simplification_patterns):
        if current_context.get('code'):
            return generate_simpler_version(current_context['code'], current_context['language'], year_level)
        return None
    
    elif any(pattern in query_lower for pattern in elaboration_patterns):
        if current_context.get('explanation'):
            return generate_detailed_explanation(current_context['explanation'], year_level)
        return None
    
    elif any(pattern in query_lower for pattern in alternative_patterns):
        if current_context.get('code'):
            return generate_alternative_solution(current_context['code'], current_context['language'], year_level)
        return None
    
    elif any(pattern in query_lower for pattern in example_patterns):
        if current_context.get('topic'):
            return generate_example(current_context['topic'], current_context['language'], year_level)
        return None
    
    return None

def update_conversation_context(user_query: str, response: str, context: dict = None):
    """Update the conversation context with new interaction."""
    conv_context = st.session_state.conversation_context
    
    # Add to chat history
    conv_context['chat_history'].append({
        'user': user_query,
        'assistant': response,
        'timestamp': time.time()
    })
    
    # Update context if provided
    if context:
        conv_context['last_code'] = context.get('code', conv_context['last_code'])
        conv_context['last_topic'] = context.get('topic', conv_context['last_topic'])
        conv_context['last_language'] = context.get('language', conv_context['last_language'])
        conv_context['last_explanation'] = context.get('explanation', conv_context['last_explanation'])
        
        # Add to context stack
        conv_context['context_stack'].append(context)
        # Keep only last 5 contexts
        if len(conv_context['context_stack']) > 5:
            conv_context['context_stack'].pop(0)

def process_query(user_query: str, year_level: str) -> str:
    try:
        logger.debug(f"Processing query: '{user_query}' for {year_level}")
        
        # First check if this is a generic interaction
        generic_intent = detect_generic_intent(user_query)
        if generic_intent['type'] and generic_intent['type'] != 'follow_up':
            response = generate_generic_response(generic_intent)
            if response:
                update_conversation_context(user_query, response)
                return response
        
        # Then check if this is a follow-up question
        if generic_intent['type'] == 'follow_up' or generic_intent['confidence'] < 0.5:
            follow_up_response = handle_follow_up_question(user_query, year_level)
            if follow_up_response:
                update_conversation_context(user_query, follow_up_response)
                return follow_up_response
        
        # Check if this is a web application request
        if any(term in user_query.lower() for term in ["web", "website", "webpage", "calculator"]):
            try:
                web_response = generate_web_app_response(user_query, year_level)
                if web_response:
                    context = {
                        'topic': 'web_development',
                        'language': 'python',
                        'code': web_response,
                        'explanation': 'Web application tutorial'
                    }
                    update_conversation_context(user_query, web_response, context)
                    return web_response
            except Exception as e:
                logger.error(f"Error generating web app response: {str(e)}")
                return "I apologize, but I encountered an error while generating the web application example. Could you try rephrasing your request or specifying which part you'd like help with first?"
        
        # Continue with regular query processing
        query_analysis = st.session_state.components['analyzer'].analyze_query(
            query=user_query,
            year_level=year_level
        )
        
        # Search for relevant resources
        matches = st.session_state.components['retriever'].search_resources(
            query=user_query,
            year_level=year_level
        )
        
        # If no matches in current year level, search other years
        if not matches:
            matches = search_resources_across_years(user_query, year_level)
        
        # Generate appropriate response
        if matches:
            response = format_matches_for_display(matches)
            context = {
                'topic': query_analysis.topic if hasattr(query_analysis, 'topic') else None,
                'language': query_analysis.language if hasattr(query_analysis, 'language') else None,
                'code': matches[0].code if matches else None,
                'explanation': matches[0].explanation if matches and hasattr(matches[0], 'explanation') else None
            }
            update_conversation_context(user_query, response, context)
            return response
        
        # If no matches found, use API response
        api_response = st.session_state.components['api_generator'].generate_response(
            query=user_query,
            year_level=year_level
        )
        
        if api_response:
            context = {
                'topic': query_analysis.topic if hasattr(query_analysis, 'topic') else None,
                'language': query_analysis.language if hasattr(query_analysis, 'language') else None,
                'explanation': api_response
            }
            update_conversation_context(user_query, api_response, context)
            return api_response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return "I apologize, but I encountered an error. Could you try rephrasing your question or breaking it down into smaller parts?"

def generate_simpler_version(code: str, language: str, year_level: str) -> str:
    """Generate a simpler version of the given code."""
    tech_level = get_technical_level(year_level)
    
    # Get language configuration
    lang_info = get_language_info(language)
    
    # Get the basic version of the code
    simple_code = lang_info["hello_world"]["basic"]
    
    response = f"""Here's a simpler version of the code:

```{language}
{simple_code}
```

This version is more straightforward because:
- It uses basic language features
- Has fewer lines of code
- Uses simpler syntax
- Is easier to understand

Would you like me to explain any part of this simpler version?"""
    
    return response

def generate_alternative_solution(code: str, language: str, year_level: str) -> str:
    """Generate an alternative solution for the given code."""
    tech_level = get_technical_level(year_level)
    
    # Get language configuration
    lang_info = get_language_info(language)
    
    # Get appropriate example based on technical level
    if tech_level["depth"] == "basic":
        alt_code = lang_info["hello_world"]["intermediate"]
    else:
        alt_code = lang_info["hello_world"]["advanced"]
    
    response = f"""Here's an alternative way to solve this:

```{language}
{alt_code}
```

This approach is different because:
- It uses a different programming paradigm
- Implements the solution using different language features
- Might be more suitable for certain use cases

Would you like me to explain the differences in detail?"""
    
    return response

def generate_example(topic: str, language: str, year_level: str) -> str:
    """Generate an example for the given topic."""
    tech_level = get_technical_level(year_level)
    
    # Get language configuration
    lang_info = get_language_info(language)
    
    # Get concepts for this level
    concepts = lang_info["concepts"][tech_level["depth"]]
    
    # Get appropriate example based on technical level
    if tech_level["depth"] == "basic":
        example_code = lang_info["hello_world"]["basic"]
    elif tech_level["depth"] == "intermediate":
        example_code = lang_info["hello_world"]["intermediate"]
    else:
        example_code = lang_info["hello_world"]["advanced"]
    
    response = f"""Here's an example that demonstrates {topic} in {language}:

```{language}
{example_code}
```

This example shows:
- How to use {', '.join(concepts)}
- Basic structure of a {language} program
- Common programming patterns

Would you like me to explain how this example works?"""
    
    return response

def get_web_framework_config(language: str) -> dict:
    """Get configuration for web frameworks based on programming language."""
    return {
        "python": {
            "frameworks": {
                "flask": {
                    "name": "Flask",
                    "setup": "pip install flask python-dotenv",
                    "imports": "from flask import Flask, render_template, request, jsonify",
                    "app_init": """app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')""",
                    "calculator_route": """@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    try:
        expression = data['expression']
        result = eval(expression)  # Note: In production, use a safer evaluation method
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400"""
                },
                "django": {
                    "name": "Django",
                    "setup": "pip install django python-dotenv",
                    "imports": """from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json""",
                    "app_init": """def index(request):
    return render(request, 'calculator/index.html')""",
                    "calculator_route": """@csrf_exempt
def calculate(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        try:
            expression = data['expression']
            result = eval(expression)  # Note: In production, use a safer evaluation method
            return JsonResponse({'result': result})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)"""
                },
                "fastapi": {
                    "name": "FastAPI",
                    "setup": "pip install fastapi uvicorn jinja2 python-dotenv",
                    "imports": """from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel""",
                    "app_init": """app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})""",
                    "calculator_route": """class Expression(BaseModel):
    expression: str

@app.post("/calculate")
async def calculate(expr: Expression):
    try:
        result = eval(expr.expression)  # Note: In production, use a safer evaluation method
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}"""
                }
            }
        },
        "javascript": {
            "frameworks": {
                "express": {
                    "name": "Express.js",
                    "setup": "npm install express body-parser",
                    "imports": """const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');""",
                    "app_init": """const app = express();
app.use(bodyParser.json());
app.use(express.static('public'));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});""",
                    "calculator_route": """app.post('/calculate', (req, res) => {
    try {
        const { expression } = req.body;
        const result = eval(expression); // Note: In production, use a safer evaluation method
        res.json({ result });
    } catch (error) {
        res.status(400).json({ error: error.message });
    }
});"""
                },
                "koa": {
                    "name": "Koa.js",
                    "setup": "npm install koa koa-router koa-static koa-body",
                    "imports": """const Koa = require('koa');
const Router = require('koa-router');
const serve = require('koa-static');
const { koaBody } = require('koa-body');""",
                    "app_init": """const app = new Koa();
const router = new Router();

app.use(koaBody());
app.use(serve('public'));

router.get('/', async (ctx) => {
    ctx.type = 'html';
    ctx.body = await fs.readFile('public/index.html');
});""",
                    "calculator_route": """router.post('/calculate', async (ctx) => {
    try {
        const { expression } = ctx.request.body;
        const result = eval(expression); // Note: In production, use a safer evaluation method
        ctx.body = { result };
    } catch (error) {
        ctx.status = 400;
        ctx.body = { error: error.message };
    }
});"""
                }
            }
        },
        "php": {
            "frameworks": {
                "laravel": {
                    "name": "Laravel",
                    "setup": "composer create-project laravel/laravel calculator-app",
                    "imports": "<?php\n\nnamespace App\\Http\\Controllers;",
                    "app_init": """class CalculatorController extends Controller
{
    public function index()
    {
        return view('calculator.index');
    }""",
                    "calculator_route": """public function calculate(Request $request)
    {
        try {
            $expression = $request->input('expression');
            $result = eval('return ' . $expression . ';');
            return response()->json(['result' => $result]);
        } catch (\\Exception $e) {
            return response()->json(['error' => $e->getMessage()], 400);
        }
    }"""
                }
            }
        }
    }

def detect_language_and_framework(query: str) -> tuple:
    """Detect programming language and framework preferences from query."""
    query_lower = query.lower()
    
    # Language detection
    languages = {
        "python": ["python", "flask", "django", "fastapi"],
        "javascript": ["javascript", "js", "node", "express", "koa"],
        "php": ["php", "laravel"]
    }
    
    detected_language = None
    for lang, keywords in languages.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_language = lang
            break
    
    # Framework detection
    frameworks = {
        "python": ["flask", "django", "fastapi"],
        "javascript": ["express", "koa"],
        "php": ["laravel"]
    }
    
    detected_framework = None
    if detected_language:
        for framework in frameworks[detected_language]:
            if framework in query_lower:
                detected_framework = framework
                break
        
        # Default frameworks if none specified
        if not detected_framework:
            defaults = {
                "python": "flask",
                "javascript": "express",
                "php": "laravel"
            }
            detected_framework = defaults[detected_language]
    
    # Default to Python/Flask if no language detected
    if not detected_language:
        detected_language = "python"
        detected_framework = "flask"
    
    return detected_language, detected_framework

def generate_web_app_response(query: str, year_level: str) -> str:
    """Generate a structured response for web application requests."""
    
    # Detect language and framework
    language, framework = detect_language_and_framework(query)
    framework_config = get_web_framework_config(language)
    
    # Get framework-specific configuration
    framework_info = framework_config[language]["frameworks"][framework]
    
    is_calculator = "calculator" in query.lower()
    needs_frontend = "frontend" in query.lower() or "webpage" in query.lower() or "website" in query.lower()
    
    if is_calculator:
        title = f"Building a Web-Based Calculator with {framework_info['name']}"
        description = f"""Let's create a calculator web application using {language.title()} with {framework_info['name']}. 
This project will teach you essential web development concepts while building something practical."""
        
        components = [
            {
                "name": f"Backend ({language.title()})",
                "description": "The server-side logic that handles calculations and routing.",
                "requirements": [
                    f"{framework_info['name']} for server implementation",
                    f"{language.title()} for backend logic",
                    "Development environment setup",
                    "Package/dependency management"
                ]
            },
            {
                "name": "Frontend (HTML/CSS/JavaScript)",
                "description": "The client-side interface that users interact with.",
                "requirements": [
                    "HTML5 for structure",
                    "CSS3 for styling",
                    "JavaScript for interactive features",
                    "Bootstrap for responsive design"
                ]
            }
        ]
        
        # Framework-specific setup steps
        setup_steps = [
            "1. Create a new project directory:\n   ```bash\n   mkdir calculator-app\n   cd calculator-app\n   ```",
            f"2. Set up the development environment:\n   ```bash\n   {framework_info['setup']}\n   ```",
            "3. Create the project structure:\n   ```\n   calculator-app/\n   ├── static/\n   │   ├── css/\n   │   │   └── style.css\n   │   └── js/\n   │       └── script.js\n   ├── templates/\n   │   └── index.html\n   ├── app.{ext}\n   └── requirements.txt\n   ```".replace("{ext}", "py" if language == "python" else "js" if language == "javascript" else "php")
        ]
        
        code_sections = [
            {
                "title": f"Backend (app.{language})",
                "language": language,
                "description": "The main application file that handles routing and calculations:",
                "code": f"""{framework_info['imports']}

{framework_info['app_init']}

{framework_info['calculator_route']}""",
                "explanation": f"""- Creates a {framework_info['name']} application instance
- Defines necessary routes and endpoints
- Handles calculation requests
- Includes error handling"""
            }
        ]
        
        # Add frontend code sections (these remain the same across frameworks)
        code_sections.extend([
            {
                "title": "Frontend (templates/index.html)",
                "language": "html",
                "description": "The calculator interface with a modern, responsive design:",
                "code": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Web Calculator</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <div class="container mt-5">
        <div class="calculator card">
            <div class="card-body">
                <input type="text" class="form-control mb-3" id="display" readonly>
                <div class="buttons">
                    <button class="btn btn-light" onclick="appendNumber('7')">7</button>
                    <button class="btn btn-light" onclick="appendNumber('8')">8</button>
                    <button class="btn btn-light" onclick="appendNumber('9')">9</button>
                    <button class="btn btn-primary" onclick="appendOperator('+')">+</button>
                    
                    <button class="btn btn-light" onclick="appendNumber('4')">4</button>
                    <button class="btn btn-light" onclick="appendNumber('5')">5</button>
                    <button class="btn btn-light" onclick="appendNumber('6')">6</button>
                    <button class="btn btn-primary" onclick="appendOperator('-')">-</button>
                    
                    <button class="btn btn-light" onclick="appendNumber('1')">1</button>
                    <button class="btn btn-light" onclick="appendNumber('2')">2</button>
                    <button class="btn btn-light" onclick="appendNumber('3')">3</button>
                    <button class="btn btn-primary" onclick="appendOperator('*')">×</button>
                    
                    <button class="btn btn-light" onclick="appendNumber('0')">0</button>
                    <button class="btn btn-light" onclick="appendNumber('.')">.</button>
                    <button class="btn btn-success" onclick="calculate()">=</button>
                    <button class="btn btn-primary" onclick="appendOperator('/')">/</button>
                    
                    <button class="btn btn-danger col-12" onclick="clearDisplay()">Clear</button>
                </div>
            </div>
        </div>
    </div>
    <script src="/static/js/script.js"></script>
</body>
</html>""",
                "explanation": """- Uses Bootstrap for responsive layout
- Implements a grid-based calculator interface
- Includes all basic calculator buttons
- Links to custom CSS and JavaScript files"""
            },
            {
                "title": "Styling (static/css/style.css)",
                "language": "css",
                "description": "Custom styles for the calculator:",
                "code": """.calculator {
    max-width: 400px;
    margin: 0 auto;
    background-color: #f8f9fa;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}

.buttons {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}

.btn {
    padding: 15px;
    font-size: 1.2rem;
}

#display {
    font-size: 1.5rem;
    text-align: right;
    background-color: white;
}""",
                "explanation": """- Creates a modern, clean calculator design
- Uses CSS Grid for button layout
- Adds subtle shadows and rounded corners
- Makes the interface responsive"""
            },
            {
                "title": "Functionality (static/js/script.js)",
                "language": "javascript",
                "description": "Calculator logic and API interaction:",
                "code": """let display = document.getElementById('display');

function appendNumber(num) {
    display.value += num;
}

function appendOperator(op) {
    display.value += op;
}

function clearDisplay() {
    display.value = '';
}

function calculate() {
    fetch('/calculate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            expression: display.value
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            display.value = 'Error';
        } else {
            display.value = data.result;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        display.value = 'Error';
    });
}""",
                "explanation": """- Handles button clicks and display updates
- Communicates with the backend API
- Includes error handling
- Uses modern fetch API for requests"""
            }
        ])
        
        next_steps = [
            "Add more advanced operations (square root, powers, etc.)",
            "Implement keyboard support",
            "Add calculation history",
            "Improve error handling and input validation",
            "Add unit tests for the calculator logic",
            "Deploy the application to a web server"
        ]
        
        return format_web_app_tutorial(
            title=title,
            description=description,
            components=components,
            setup_steps=setup_steps,
            code_sections=code_sections,
            next_steps=next_steps
        )
    
    return None

def format_web_app_tutorial(title: str, description: str, components: List[dict], setup_steps: List[str], code_sections: List[dict], next_steps: List[str]) -> str:
    """Format a web application tutorial response in a clear, structured way."""
    
    # Format the main sections
    response_parts = [
        f"# {title}\n",
        f"{description}\n",
        "\n## Project Components",
    ]
    
    # Add components section
    for component in components:
        response_parts.append(f"### {component['name']}")
        response_parts.append(component['description'])
        if 'requirements' in component:
            response_parts.append("\nRequirements:")
            for req in component['requirements']:
                response_parts.append(f"- {req}")
        response_parts.append("")
    
    # Add setup section
    response_parts.extend([
        "\n## Setup Instructions",
        "\nFollow these steps to set up your development environment:\n"
    ])
    for step in setup_steps:
        response_parts.append(f"{step}\n")
    
    # Add code sections with explanations
    response_parts.append("\n## Implementation")
    for section in code_sections:
        response_parts.extend([
            f"\n### {section['title']}",
            section['description'],
            f"\n```{section['language']}",
            section['code'],
            "```\n",
            "**Explanation:**",
            section['explanation'],
            "\n"
        ])
    
    # Add next steps
    response_parts.extend([
        "\n## Next Steps",
        "\nTo enhance your application, you could:",
    ])
    for step in next_steps:
        response_parts.append(f"- {step}")
    
    # Add final note
    response_parts.append("\nWould you like me to explain any part in more detail?")
    
    return "\n".join(response_parts)

# Main chat interface
def main():
    # Add custom CSS for dark theme and layout
    st.markdown("""
        <style>
        /* Reset and base styles */
        .stApp {
            background-color: #0E1117;
        }
        
        .main-title {
            color: white;
            font-size: 2.5rem !important;
            padding: 1rem 0;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Sidebar customization */
        .css-1d391kg {
            background-color: #1A1E24;
            padding: 2rem 1rem;
        }
        
        /* Chat container */
        .stChatMessage {
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        /* Input field */
        .stChatInputContainer {
            background-color: #0E1117;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1rem;
        }
        
        .stChatInput {
            border-radius: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Year level selector */
        .stSelectbox {
            background-color: #1E1E1E;
        }
        
        /* Custom emoji style */
        .title-emoji {
            font-size: 1.8rem;
            transform: rotate(-15deg);
            display: inline-block;
            color: #FF9D00;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create title with emoji
    st.markdown('<h1 class="main-title">IntelliSE <span class="title-emoji">📱</span></h1>', unsafe_allow_html=True)

    # Initialize session state for messages with welcome message
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you with your coding questions today?"}
        ]
    
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = {
            'last_code': None,
            'last_topic': None,
            'last_language': None,
            'last_explanation': None,
            'chat_history': [],
            'context_stack': []
        }

    # Sidebar content
    with st.sidebar:
        st.markdown("### Year Level Selection")
        st.markdown("""
            Choose your year level to get explanations with appropriate terminology.
            
            - Year 1: Basic, beginner-friendly terms
            - Year 2: More technical terminology
            - Year 3: Professional concepts
            - Year 4: Advanced academic terms
            
            Note: You can ask about any programming language regardless of your year level!
        """)
        
        year_level = st.selectbox(
            "Select Your Year Level:",
            [
                "Year 1 Certificate",
                "Year 2 Diploma",
                "Year 3 Degree",
                "Year 4 Postgraduate Diploma"
            ],
            key="year_level"
        )

        if 'last_response_source' in st.session_state:
            source = st.session_state.last_response_source
            if source == 'local':
                st.success("🎯 Using Local Resources")
            else:
                st.info("🌐 Using External API")

    # Main chat area
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)
    
    # Chat input
    if user_query := st.chat_input("Ask your coding question..."):
        # Add user message to chat
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display thinking message
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("🤔 Thinking...")
            
            try:
                # Process query and generate response
                response = process_query(user_query, year_level)
                
                # Update display with response
                thinking_placeholder.empty()
                st.markdown(response, unsafe_allow_html=True)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                logger.error(f"Error in main chat interface: {str(e)}")
                thinking_placeholder.empty()
                error_message = "I apologize, but I encountered an error. Could you try asking your question in a different way?"
                st.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    logger.debug("Starting Coding Assistant Chatbot")
    main()