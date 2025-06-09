import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import google.generativeai as genai
from retrievalAlgorithms.resource_retriever import ResourceMatch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

@dataclass
class GeneratedResponse:
    content: str
    source_type: str  # 'local' or 'external'
    resources_used: List[str]
    confidence_score: float

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
        if not code:
            return 0.0
        
        score = 0.0
        total_factors = 5
        
        # Check for comments
        comment_ratio = len(re.findall(self.code_patterns['comments'], code)) / len(code.split('\n'))
        score += min(comment_ratio * 2, 1.0)  # Max 1 point for comments
        
        # Check for proper function/class structure
        has_functions = bool(re.search(self.code_patterns['functions'], code))
        has_classes = bool(re.search(self.code_patterns['classes'], code))
        score += (has_functions + has_classes) / 2  # Max 1 point for structure
        
        # Check for variable naming (basic check for descriptive names)
        variables = re.findall(self.code_patterns['variables'], code)
        if variables:
            avg_var_length = sum(len(var.split()[-1]) for var in variables) / len(variables)
            score += min(avg_var_length / 10, 1.0)  # Max 1 point for naming
        
        # Check for error handling
        has_error_handling = bool(re.search(self.code_patterns['error_handling'], code))
        score += 1.0 if has_error_handling else 0.0  # 1 point for error handling
        
        # Check for consistent indentation
        lines = code.split('\n')
        indentation_patterns = [line[:len(line) - len(line.lstrip())] for line in lines if line.strip()]
        if indentation_patterns:
            unique_patterns = len(set(indentation_patterns))
            score += 1.0 if unique_patterns <= 3 else 0.5  # Max 1 point for consistent indentation
        
        return score / total_factors

    def _filter_and_combine_matches(self, matches: List[ResourceMatch]) -> List[ResourceMatch]:
        """Filter and combine the most relevant matches."""
        if not matches:
            return []
        
        # Sort by relevance score
        sorted_matches = sorted(matches, key=lambda x: x.relevance_score, reverse=True)
        
        # Take top matches
        top_matches = sorted_matches[:3]
        
        # Combine similar matches
        combined = []
        for match in top_matches:
            # Check if similar to any existing combined match
            similar_found = False
            for existing in combined:
                if self._calculate_similarity(match.content, existing.content) > 0.8:
                    # Merge information
                    existing.context += f"\n{match.context}"
                    similar_found = True
                    break
            
            if not similar_found:
                combined.append(match)
        
        return combined

    def _generate_dynamic_intro(self, query: str, query_analysis) -> str:
        """Generate a dynamic introduction based on query type and context."""
        intro = "# Response Summary\n\n"
        
        # Add context about the query type
        if query_analysis.query_type == 'explanation':
            intro += "I'll provide a detailed explanation of the concept"
        elif query_analysis.query_type == 'code':
            intro += "I'll show you practical code examples"
        else:
            intro += "I'll provide both explanation and code examples"
        
        # Add year level context
        intro += f" suitable for {query_analysis.year_level} students.\n"
        
        # Add topic category if available
        if query_analysis.topic_category != 'general':
            intro += f"\nThis relates to {query_analysis.topic_category}"
            if query_analysis.code_language:
                intro += f" in {query_analysis.code_language}"
            intro += ".\n"
        
        return intro

    def _generate_conceptual_explanation(self, matches: List[ResourceMatch]) -> str:
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

    def _generate_code_section(self, matches: List[ResourceMatch], query_analysis) -> str:
        """Generate an enhanced code section with examples and explanations."""
        code_sections = []
        
        for i, match in enumerate(matches):
            section = f"\n### Example {i+1}: {match.title}\n"
            
            # Add purpose and context
            section += f"\nPurpose: {match.purpose}\n"
            if match.context:
                section += f"Context: {match.context}\n"
            
            # Format and comment code
            enhanced_code = self._enhance_code_comments(match.code, query_analysis.year_level)
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
                    enhanced.extend(self._comment_block(current_block, year_level))
                    current_block = []
                in_function = True
                enhanced.append(line)
            
            # Group related lines
            elif in_function and stripped:
                current_block.append(line)
            
            # Handle block endings
            elif in_function and not stripped:
                if current_block:
                    enhanced.extend(self._comment_block(current_block, year_level))
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
            practices.append(f"When working with {query_analysis.code_language}:")
            # Add language-specific best practices
            lang_practices = self._get_language_best_practices(query_analysis.code_language)
            practices.extend(f"- {practice}" for practice in lang_practices)
        
        if query_analysis.topic_category != 'general':
            practices.append(f"\nFor {query_analysis.topic_category}:")
            # Add topic-specific best practices
            topic_practices = self._get_topic_best_practices(query_analysis.topic_category)
            practices.extend(f"- {practice}" for practice in topic_practices)
        
        return '\n'.join(practices)

    def _generate_practice_section(self, query_analysis) -> str:
        """Generate practice exercises or challenges."""
        exercises = ["\n## Practice Exercises\n"]
        
        # Generate exercises based on topic and difficulty
        if query_analysis.topic_category != 'general':
            exercises.append("Try these exercises to reinforce your understanding:")
            practice_problems = self._generate_practice_problems(
                query_analysis.topic_category,
                query_analysis.year_level,
                query_analysis.code_language
            )
            exercises.extend(f"\n{i+1}. {problem}" for i, problem in enumerate(practice_problems))
        
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

    def generate_response(self, 
                         query: str, 
                         matches: List[ResourceMatch], 
                         query_analysis) -> GeneratedResponse:
        """Generate an enhanced response using local resources."""
        try:
            if not matches:
                return self._generate_fallback_response(query, query_analysis)

            # Filter and prepare matches
            best_matches = self._filter_and_combine_matches(matches)
            
            # Build response sections
            sections = []
            
            # Dynamic introduction
            intro = self._generate_dynamic_intro(query, query_analysis)
            sections.append(intro)
            
            # Main content sections
            if query_analysis.query_type in ['explanation', 'both']:
                concept_explanation = self._generate_conceptual_explanation(best_matches)
                sections.append(concept_explanation)
            
            if query_analysis.query_type in ['code', 'both']:
                code_section = self._generate_code_section(best_matches, query_analysis)
                sections.append(code_section)
            
            # Best practices and tips
            practices = self._generate_best_practices(query_analysis)
            sections.append(practices)
            
            # Practice exercises
            if query_analysis.query_type != 'explanation':
                practice = self._generate_practice_section(query_analysis)
                sections.append(practice)
            
            # Additional resources
            resources = self._suggest_additional_resources(query_analysis)
            sections.append(resources)
            
            return GeneratedResponse(
                content='\n\n'.join(sections),
                source_type='local',
                resources_used=[m.source_file for m in best_matches],
                confidence_score=0.95
            )
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return self._generate_fallback_response(query, query_analysis)

    def _generate_fallback_response(self, query: str, query_analysis) -> GeneratedResponse:
        """Generate a fallback response when no matches are found."""
        content = [
            "# I apologize, but I couldn't find specific information about that.",
            "\nHowever, here are some general resources that might help:"
        ]
        
        # Add general best practices
        if query_analysis.code_language:
            practices = self._get_language_best_practices(query_analysis.code_language)
            content.append("\n## Best Practices:")
            content.extend(f"- {practice}" for practice in practices)
        
        # Add documentation links
        if query_analysis.code_language:
            docs = self._get_documentation_links(query_analysis.code_language)
            content.append("\n## Helpful Documentation:")
            content.extend(f"- {doc}" for doc in docs)
        
        return GeneratedResponse(
            content='\n'.join(content),
            source_type='local',
            resources_used=[],
            confidence_score=0.3
        )

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            # Encode texts
            embedding1 = self.sentence_encoder.encode([text1])[0]
            embedding2 = self.sentence_encoder.encode([text2])[0]
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0

class APIResponseGenerator:
    """Handles response generation using external API when local resources are insufficient."""
    
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=api_key)
        
        # Configure the model
        generation_config = {
            'temperature': 0.7,
            'top_p': 1,
            'top_k': 1,
            'max_output_tokens': 2048,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        self.model = genai.GenerativeModel(
            model_name='gemini-pro',
            generation_config=generation_config,
            safety_settings=safety_settings
        )

    def generate_response(self, query: str, query_analysis) -> GeneratedResponse:
        """Generate a response using the external API."""
        try:
            # Prepare prompt
            prompt = self._prepare_prompt(query, query_analysis)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            return GeneratedResponse(
                content=response.text,
                source_type='external',
                resources_used=['Google Gemini API'],
                confidence_score=0.8
            )
            
        except Exception as e:
            print(f"API Error: {str(e)}")
            return GeneratedResponse(
                content="I apologize, but I encountered an error while generating the response. "
                       "Please try again or rephrase your question.",
                source_type='external',
                resources_used=[],
                confidence_score=0.0
            )

    def _prepare_prompt(self, query: str, query_analysis) -> str:
        """Prepare a detailed prompt for the API."""
        prompt = [
            f"You are a computer science tutor helping a {query_analysis.year_level} student.",
            f"The student's question is: {query}",
            "\nProvide a comprehensive response that includes:",
            "1. A clear explanation of the concept",
            "2. Practical code examples if relevant",
            "3. Best practices and common pitfalls",
            "4. Additional resources for learning",
            "\nFormat the response in Markdown with appropriate sections and code blocks."
        ]
        
        if query_analysis.code_language:
            prompt.append(f"\nUse {query_analysis.code_language} for any code examples.")
        
        if query_analysis.topic_category != 'general':
            prompt.append(f"\nFocus on {query_analysis.topic_category} concepts.")
        
        return '\n'.join(prompt)

def format_response_for_display(generated_response: GeneratedResponse) -> str:
    """Format the response for display in the UI."""
    formatted = [
        generated_response.content,
        "\n---\n",
        f"Source: {'External API' if generated_response.source_type == 'external' else 'Local Resources'}",
    ]
    
    if generated_response.resources_used:
        formatted.append("\nResources Used:")
        formatted.extend(f"- {resource}" for resource in generated_response.resources_used)
    
    formatted.append(f"\nConfidence Score: {generated_response.confidence_score:.2f}")
    
    return '\n'.join(formatted) 