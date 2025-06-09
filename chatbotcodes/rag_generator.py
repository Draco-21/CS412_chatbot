import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import logging
from .models import QueryAnalysis, ResourceMatch, GeneratedResponse

logger = logging.getLogger(__name__)

class LocalResponseGenerator:
    """Enhanced response generation using local resources with advanced NLP techniques."""

    def __init__(self):
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.code_patterns = {
            'comments': r'(#.*|//.*|/\*[\s\S]*?\*/)',
            'functions': r'(def\s+\w+|function\s+\w+|\w+\s*=\s*function)',
            'classes': r'class\s+\w+',
            'variables': r'(let|var|const|int|float|string|bool)\s+\w+',
            'error_handling': r'(try|catch|except|finally|raise|throw)',
        }

        # Initialize best practices database
        self.language_best_practices = {
            'python': [
                "Use meaningful variable and function names",
                "Follow PEP 8 style guidelines",
                "Use list comprehensions for simple loops",
                "Utilize built-in functions and libraries",
                "Handle exceptions appropriately"
            ],
            'java': [
                "Follow Java naming conventions",
                "Use proper access modifiers",
                "Implement interfaces where appropriate",
                "Handle exceptions with try-catch blocks",
                "Use StringBuilder for string concatenation"
            ],
            'cpp': [
                "Use references instead of pointers when possible",
                "Follow RAII principles",
                "Use const correctly",
                "Prefer std algorithms over raw loops",
                "Use smart pointers instead of raw pointers"
            ],
            'javascript': [
                "Use const and let instead of var",
                "Implement proper error handling",
                "Use modern ES6+ features",
                "Follow asynchronous programming best practices",
                "Use strict mode"
            ]
        }

        self.topic_best_practices = {
            'algorithm': [
                "Analyze time and space complexity",
                "Consider edge cases",
                "Test with different input sizes",
                "Optimize for readability first",
                "Document your approach"
            ],
            'data_structure': [
                "Choose appropriate data structures",
                "Consider memory usage",
                "Implement standard operations",
                "Handle edge cases",
                "Document usage examples"
            ],
            'web_dev': [
                "Follow responsive design principles",
                "Implement proper error handling",
                "Use semantic HTML",
                "Consider accessibility",
                "Follow security best practices"
            ],
            'database': [
                "Use proper indexing",
                "Write optimized queries",
                "Handle transactions properly",
                "Implement data validation",
                "Follow normalization rules"
            ]
        }

        # Initialize documentation links
        self.documentation_links = {
            'python': [
                "Python Official Documentation: https://docs.python.org/3/",
                "Python Package Index: https://pypi.org/",
                "Real Python Tutorials: https://realpython.com/"
            ],
            'java': [
                "Java SE Documentation: https://docs.oracle.com/en/java/",
                "Spring Framework: https://spring.io/guides",
                "Java Tutorial Point: https://www.tutorialspoint.com/java/"
            ],
            'cpp': [
                "C++ Reference: https://en.cppreference.com/",
                "C++ Tutorial: https://www.learncpp.com/",
                "Modern C++ Features: https://github.com/AnthonyCalandra/modern-cpp-features"
            ],
            'javascript': [
                "MDN Web Docs: https://developer.mozilla.org/en-US/docs/Web/JavaScript",
                "JavaScript.info: https://javascript.info/",
                "Node.js Documentation: https://nodejs.org/en/docs/"
            ]
        }

    def _assess_code_quality(self, code: str) -> float:
        """Evaluate code quality based on multiple factors."""
        score = 0.0

        # Check for comments
        comments = re.findall(self.code_patterns['comments'], code)
        comment_ratio = len(comments) / max(len(code.split('\n')), 1)
        score += min(0.3, comment_ratio)

        # Check for proper structure
        has_functions = bool(re.search(self.code_patterns['functions'], code))
        has_classes = bool(re.search(self.code_patterns['classes'], code))
        has_error_handling = bool(
            re.search(
                self.code_patterns['error_handling'],
                code))

        structure_score = (has_functions + has_classes +
                           has_error_handling) / 3
        score += 0.3 * structure_score

        # Check indentation consistency
        lines = code.split('\n')
        indentation_patterns = [len(line) - len(line.lstrip())
                                for line in lines if line.strip()]
        if indentation_patterns:
            consistency = 1 - (len(set(indentation_patterns)
                                   ) / len(indentation_patterns))
            score += 0.2 * consistency

        # Check variable naming
        variables = re.findall(r'\b[a-zA-Z_]\w*\b', code)
        meaningful_names = sum(1 for var in variables if len(var) > 1)
        score += 0.2 * (meaningful_names / max(len(variables), 1))

        return min(score, 1.0)

    def _filter_and_combine_matches(
            self, matches: List[ResourceMatch]) -> List[ResourceMatch]:
        """Smart filtering and combination of matches."""
        if not matches:
            return []

        filtered = []
        covered_topics = set()

        for match in matches:
            # Check if this match adds new information
            match_topics = set(match.topics)
            if not match_topics.issubset(
                    covered_topics) and match.score >= 0.3:
                filtered.append(match)
                covered_topics.update(match_topics)

            if len(filtered) >= 3:  # Limit to top 3 diverse matches
                break

        return filtered

    def _generate_dynamic_intro(self, query: str, query_analysis) -> str:
        """Generate a context-aware introduction."""
        intro_parts = []

        # Base introduction
        intro_parts.append(
            f"Based on the {
                query_analysis.year_level} curriculum, let me help you understand")

        # Add topic focus
        if query_analysis.topic_category != 'general':
            intro_parts.append(f"this {query_analysis.topic_category} concept")

        # Add language/framework context
        if query_analysis.code_language:
            intro_parts.append(f"using {query_analysis.code_language}")
            if query_analysis.frameworks:
                intro_parts.append(
                    f"with {
                        ', '.join(
                            query_analysis.frameworks)}")

        return ' '.join(intro_parts) + '.'

    def _generate_conceptual_explanation(
            self, matches: List[ResourceMatch]) -> str:
        """Generate a comprehensive conceptual explanation."""
        explanations = []
        
        for match in matches:
            # Extract key concepts
            concepts = self._extract_key_concepts(match.purpose, match.context)
            
            # Format explanation
            explanation = f"\n## {match.title}\n"
            explanation += f"\n{match.purpose}\n"
            
            # Add key concepts
            if concepts:
                explanation += "\nKey Concepts:\n"
                for concept in concepts:
                    explanation += f"- {concept}\n"
            
            explanations.append(explanation)
        
        return '\n'.join(explanations)

    def _generate_code_section(
            self, matches: List[ResourceMatch], query_analysis) -> str:
        """Generate an enhanced code section with examples and explanations."""
        code_sections = []

        for i, match in matches:
            section = f"\n### Example {i + 1}: {match.title}\n"

            # Add purpose and context
            section += f"\nPurpose: {match.purpose}\n"
            if match.context:
                section += f"Context: {match.context}\n"

            # Format and comment code
            enhanced_code = self._enhance_code_comments(
                match.code, query_analysis.year_level)
            section += f"\n```{match.language}\n{enhanced_code}\n```\n"

            code_sections.append(section)

        return '\n'.join(code_sections)

    def _enhance_code_comments(self, code: str, year_level: str) -> str:
        """Enhance code with detailed comments based on complexity level."""
        lines = code.split('\n')
        enhanced = []

        in_function = False
        current_block = []
        
        for line in lines:
            stripped = line.strip()

            # Detect function/class definitions
            if re.match(r'(def|class)\s+\w+', stripped):
                if current_block:
                    enhanced.extend(
                        self._comment_block(
                            current_block, year_level))
                    current_block = []
                in_function = True
                enhanced.append(line)

            # Group related lines
            elif in_function and stripped:
                current_block.append(line)
        
            # Handle block endings
            elif in_function and not stripped:
                if current_block:
                    enhanced.extend(
                        self._comment_block(
                            current_block, year_level))
                    current_block = []
                enhanced.append(line)
                in_function = False
            else:
                enhanced.append(line)

        # Handle any remaining block
        if current_block:
            enhanced.extend(self._comment_block(current_block, year_level))

        return '\n'.join(enhanced)

    def _comment_block(self, block: List[str], year_level: str) -> List[str]:
        """Add appropriate comments to a block of code based on year level."""
        commented = []

        # Determine comment detail level
        detail_level = 'basic' if 'Year 1' in year_level else 'intermediate'

        # Add block-level comment
        block_purpose = self._infer_block_purpose(block)
        if block_purpose:
            commented.append(f"# {block_purpose}")

        # Add line-level comments based on detail level
        for line in block:
            if detail_level == 'basic' and self._is_important_concept(line):
                commented.append(f"# Explanation: {self._explain_line(line)}")
            commented.append(line)

        return commented

    def _generate_best_practices(self, query_analysis) -> str:
        """Generate relevant best practices and tips."""
        practices = ["\n## Best Practices and Tips\n"]

        if query_analysis.code_language:
            practices.append(
                f"When working with {
                    query_analysis.code_language}:")
            # Add language-specific best practices
            lang_practices = self._get_language_best_practices(
                query_analysis.code_language)
            practices.extend(f"- {practice}" for practice in lang_practices)

        if query_analysis.topic_category != 'general':
            practices.append(f"\nFor {query_analysis.topic_category}:")
            # Add topic-specific best practices
            topic_practices = self._get_topic_best_practices(
                query_analysis.topic_category)
            practices.extend(f"- {practice}" for practice in topic_practices)

        return '\n'.join(practices)

    def _generate_practice_section(self, query_analysis) -> str:
        """Generate practice exercises or challenges."""
        exercises = ["\n## Practice Exercises\n"]

        # Generate exercises based on topic and difficulty
        if query_analysis.topic_category != 'general':
            exercises.append(
                "Try these exercises to reinforce your understanding:")
            practice_problems = self._generate_practice_problems(
                query_analysis.topic_category,
                query_analysis.year_level,
                query_analysis.code_language
            )
            exercises.extend(
                f"\n{
                    i + 1}. {problem}" for i,
                problem in enumerate(practice_problems))

        return '\n'.join(exercises)

    def _suggest_additional_resources(self, query_analysis) -> str:
        """Suggest additional learning resources."""
        resources = ["\n## Additional Resources\n"]

        # Add relevant documentation
        if query_analysis.code_language:
            docs = self._get_documentation_links(query_analysis.code_language)
            resources.append("Official Documentation:")
            resources.extend(f"- {doc}" for doc in docs)

        # Add tutorial recommendations
        tutorials = self._get_tutorial_recommendations(
            query_analysis.topic_category,
            query_analysis.year_level
        )
        if tutorials:
            resources.append("\nRecommended Tutorials:")
            resources.extend(f"- {tutorial}" for tutorial in tutorials)

        return '\n'.join(resources)

    def generate_response(self, query: str, matches: List[ResourceMatch], analysis: QueryAnalysis) -> str:
        """Generate response using local resources and algorithm knowledge."""
        
        # If it's an algorithm query and we have specific information
        if analysis.topic_category == 'algorithm' and analysis.algorithm_type:
            logger.debug(f"Generating algorithm response for: {analysis.algorithm_type}")
            return generate_algorithm_response(
                analysis.algorithm_type,
                analysis.algorithm_complexity,
                analysis.code_language
            )
            
        # Continue with existing response generation for non-algorithm queries
        if not matches:
            return "I apologize, but I couldn't find any relevant information for your query."
            
        # Format matches into a response
        # First, create a GeneratedResponse object from your matches and analysis
        # This is a conceptual placeholder; you'll need to fill in the actual logic
        # based on how you want to combine the information from `matches`.
        # For example, you might concatenate the content from each ResourceMatch.
        combined_content = ""
        used_resources = []
        for match in matches:
            combined_content += f"\n## {match.title}\n"
            combined_content += f"{match.purpose}\n"
            if match.code:
                combined_content += f"```{match.language}\n{match.code}\n```\n"
            if match.url:
                combined_content += f"Source: {match.url}\n"
            used_resources.append(match.title) # Or a more descriptive identifier

        # Assuming you want to indicate a local source and some confidence
        generated_res_obj = GeneratedResponse(
            content=combined_content,
            source_type='local',
            resources_used=used_resources,
            confidence_score=0.8 # You can define a logic for this
        )

        response = format_response_for_display(generated_res_obj)
        
        return response

    def _extract_key_concepts(self, purpose: str, context: str) -> List[str]:
        """Extract key concepts from purpose and context."""
        text = f"{purpose} {context}"

        # Extract phrases that look like concepts
        concept_patterns = [
            r'(?:using|with|through) ([A-Za-z\s]+)',
            r'(?:concept of|about) ([A-Za-z\s]+)',
            r'([A-Za-z\s]+) (?:algorithm|approach|method)',
            r'([A-Za-z\s]+) (?:structure|pattern|principle)'
        ]

        concepts = []
        for pattern in concept_patterns:
            matches = re.finditer(pattern, text)
            concepts.extend(match.group(1).strip() for match in matches)

        # Remove duplicates and filter out short concepts
        concepts = list(
            set(concept for concept in concepts if len(concept.split()) >= 2))
        return concepts[:5]  # Return top 5 concepts

    def _infer_block_purpose(self, block: List[str]) -> Optional[str]:
        """Infer the purpose of a code block."""
        block_text = '\n'.join(block)

        # Check for common patterns
        if re.search(r'(class|struct)\s+\w+', block_text):
            return "Class definition for object structure"
        elif re.search(r'def\s+\w+\s*\([^)]*\)', block_text):
            return "Function definition for specific operation"
        elif re.search(r'(if|else|elif)\s+', block_text):
            return "Conditional logic for decision making"
        elif re.search(r'(for|while)\s+', block_text):
            return "Loop for iterative operation"
        elif re.search(r'try\s*:', block_text):
            return "Error handling block"
        elif re.search(r'return\s+', block_text):
            return "Return statement with computation result"

        return None

    def _is_important_concept(self, line: str) -> bool:
        """Determine if a line contains an important programming concept."""
        important_patterns = [
            r'\b(if|else|elif|for|while|try|except|class|def)\b',
            r'\b(return|yield|raise|with|import|from)\b',
            r'\b(list|dict|set|tuple|array|vector)\b',
            r'\b(function|method|property|static|private|public)\b'
        ]

        return any(re.search(pattern, line) for pattern in important_patterns)

    def _explain_line(self, line: str) -> str:
        """Generate a simple explanation for a code line."""
        line = line.strip()

        # Variable assignment
        if '=' in line and not line.startswith(('if', 'while')):
            return "Assigns a value to a variable"

        # Control structures
        if line.startswith('if'):
            return "Checks a condition to make a decision"
        if line.startswith('for'):
            return "Starts a loop to repeat operations"
        if line.startswith('while'):
            return "Begins a loop that continues while a condition is true"

        # Function definition
        if line.startswith('def'):
            return "Defines a new function"

        # Class definition
        if line.startswith('class'):
            return "Defines a new class (object template)"

        # Return statement
        if line.startswith('return'):
            return "Returns a value from the function"

        return "Performs an operation"

    def _get_language_best_practices(self, language: str) -> List[str]:
        """Get best practices for a specific programming language."""
        return self.language_best_practices.get(language.lower(), [
            "Use clear and descriptive names",
            "Comment your code appropriately",
            "Follow language conventions",
            "Handle errors properly",
            "Write modular code"
        ])

    def _get_topic_best_practices(self, topic: str) -> List[str]:
        """Get best practices for a specific topic."""
        return self.topic_best_practices.get(topic.lower(), [
            "Start with simple examples",
            "Break down complex problems",
            "Test thoroughly",
            "Document your approach",
            "Consider performance implications"
        ])

    def _get_documentation_links(self, language: str) -> List[str]:
        """Get relevant documentation links for a programming language."""
        return self.documentation_links.get(language.lower(), [
            "Search for official documentation",
            "Look for community tutorials",
            "Check popular learning platforms"
        ])

    def _get_tutorial_recommendations(
            self, topic: str, year_level: str) -> List[str]:
        """Generate tutorial recommendations based on topic and year level."""
        difficulty = "beginner" if "Year 1" in year_level else "intermediate"

        recommendations = [
            f"Search for {difficulty} tutorials on {topic}",
            f"Look for {difficulty} projects related to {topic}",
            f"Find {difficulty} exercises about {topic}"
        ]

        return recommendations

    def _generate_practice_problems(
            self, topic: str, year_level: str, language: str) -> List[str]:
        """Generate practice problems based on topic and difficulty."""
        is_beginner = "Year 1" in year_level

        if topic == 'algorithm':
            if is_beginner:
                return [
                    "Write a function to find the maximum element in an array",
                    "Implement a simple sorting algorithm",
                    "Create a function to check if a string is palindrome"
                ]
            else:
                return [
                    "Implement a divide-and-conquer algorithm",
                    "Create an efficient searching algorithm",
                    "Solve a dynamic programming problem"
                ]

        elif topic == 'data_structure':
            if is_beginner:
                return [
                    "Implement a simple linked list",
                    "Create a basic stack implementation",
                    "Build a queue using arrays"
                ]
            else:
                return [
                    "Implement a balanced binary tree",
                    "Create a hash table with collision handling",
                    "Build a priority queue"
                ]

        elif topic == 'web_dev':
            if is_beginner:
                return [
                    "Create a simple responsive webpage",
                    "Build a basic form with validation",
                    "Implement a navigation menu"
                ]
            else:
                return [
                    "Build a RESTful API",
                    "Implement authentication system",
                    "Create a dynamic web application"
                ]

        # Default problems
        if is_beginner:
            return [
                "Write a program to solve a simple problem",
                "Implement basic functionality",
                "Create a small utility function"
            ]
        else:
            return [
                "Implement a complex feature",
                "Create an efficient solution",
                "Build a complete system"
            ]


class APIResponseGenerator:
    """Handles response generation using external API when local resources are insufficient."""
    
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )

    def generate_response(self, query: str,
                          query_analysis) -> GeneratedResponse:
        """Generate a response using the external API."""
        try:
            prompt = f"""You are an educational coding assistant helping a {query_analysis.year_level} student.

Student's Question: {query}

Please provide a response that:
1. Is appropriate for their academic level
2. Includes practical code examples
3. Focuses on {query_analysis.topic_category} concepts
4. Uses {query_analysis.code_language or 'appropriate'} programming language
5. Incorporates {', '.join(query_analysis.frameworks) if query_analysis.frameworks else 'relevant'} frameworks

Response Format:
1. Start with a brief introduction
2. Provide the main explanation/answer
3. Include code examples with step-by-step explanations
4. Add any necessary clarifications
5. End with a summary or key takeaways"""

            response = self.model.generate_content(prompt)
            
            return GeneratedResponse(
                content=response.text,
                source_type='external',
                resources_used=[],
                confidence_score=0.0
            )

        except Exception as e:
            print(f"Error generating API response: {e}")
            return GeneratedResponse(
                content=f"Error generating response from API: {str(e)}",
                source_type='error',
                resources_used=[],
                confidence_score=0.0
            )


def format_response_for_display(generated_response: GeneratedResponse) -> str:
    """Format the generated response for display to the user."""
    response = ""
    
    # Add source information
    if generated_response.source_type == 'local':
        response += "📚 Response based on course materials:\n"
        if generated_response.resources_used:
            response += "Sources: " + \
                ", ".join(generated_response.resources_used) + "\n"
    elif generated_response.source_type == 'external':
        response += "🌐 Response based on general knowledge:\n"
    elif generated_response.source_type == 'error':
        response += "⚠️ Error in response generation:\n"
    
    # Add confidence score if available
    if generated_response.confidence_score > 0:
        response += f"Confidence: {generated_response.confidence_score:.2f}\n"
    
    # Add main content
    response += "\n" + generated_response.content
    
    return response 

def generate_algorithm_response(algorithm_type: str, complexity_info: dict, language: Optional[str] = None) -> str:
    """Generate a detailed response for algorithm queries."""
    if algorithm_type == 'merge_sort':
        explanation = """Merge Sort is a divide-and-conquer sorting algorithm that works as follows:

1. **Divide**: The algorithm starts by dividing the input array into two halves, then recursively divides each half until we have single elements.

2. **Conquer**: The algorithm then merges these smaller sorted arrays back together to form a larger sorted array.

3. **Key Characteristics**:
   - Time Complexity: O(n log n) - very efficient for large datasets
   - Space Complexity: O(n) - requires additional space
   - Stable Sort: Yes - maintains relative order of equal elements
   - Recursive: Yes

4. **Advantages**:
   - Predictable performance O(n log n) in all cases
   - Stable sorting algorithm
   - Efficient for large datasets

5. **Disadvantages**:
   - Requires extra space O(n)
   - Overkill for small arrays"""

        if language == 'cpp':
            code = """#include <iostream>
using namespace std;

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    // Create temp arrays
    int* L = new int[n1];
    int* R = new int[n2];
    
    // Copy data to temp arrays
    for(int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for(int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];
    
    // Merge the temp arrays back
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    
    // Copy remaining elements
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
    
    delete[] L;
    delete[] R;
}

void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        
        // Sort first and second halves
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        
        // Merge the sorted halves
        merge(arr, left, mid, right);
    }
}

int main() {
    int arr[] = {12, 11, 13, 5, 6, 7};
    int n = sizeof(arr)/sizeof(arr[0]);
    
    cout << "Original array: ";
    for(int i = 0; i < n; i++)
        cout << arr[i] << " ";
    cout << endl;
    
    mergeSort(arr, 0, n-1);
    
    cout << "Sorted array: ";
    for(int i = 0; i < n; i++)
        cout << arr[i] << " ";
    cout << endl;
    
    return 0;
}"""
            return f"{explanation}\n\n**Implementation in C++:**\n```cpp\n{code}\n```"
        else:
            return explanation

    # Add more algorithm implementations as needed
    return "Algorithm explanation not available yet."
