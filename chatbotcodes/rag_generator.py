import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import google.generativeai as genai
from retrievalAlgorithms.resource_retriever import ResourceMatch
from dotenv import load_dotenv

@dataclass
class GeneratedResponse:
    content: str
    source_type: str  # 'local' or 'external'
    resources_used: List[str]
    confidence_score: float

class LocalResponseGenerator:
    """Handles response generation using only local resources without any external API."""
    
    def _combine_code_snippets(self, matches: List[ResourceMatch]) -> str:
        """Combine multiple code snippets with proper context."""
        combined = []
        for match in matches:
            snippet = f"// From: {match.title}\n"
            snippet += f"// Purpose: {match.purpose}\n"
            snippet += match.code_snippet
            combined.append(snippet)
        return "\n\n".join(combined)

    def _generate_explanation(self, matches: List[ResourceMatch], query_type: str) -> str:
        """Generate explanation text from matches based on query type."""
        explanation = []
        
        if query_type in ['explanation', 'both']:
            # Add conceptual explanations
            for match in matches:
                explanation.append(f"• {match.purpose}")
        
        if query_type in ['code', 'both']:
            # Add code-specific explanations
            for match in matches:
                if match.framework_name:
                    explanation.append(f"• Using {match.framework_name}: {match.title}")
                else:
                    explanation.append(f"• Implementation: {match.title}")
        
        return "\n".join(explanation)

    def _format_code_walkthrough(self, match: ResourceMatch) -> str:
        """Create a step-by-step walkthrough of the code."""
        lines = match.code_snippet.split('\n')
        walkthrough = []
        current_block = []
        
        for line in lines:
            if line.strip().startswith(('//', '#')):  # Comments indicate block boundaries
                if current_block:
                    walkthrough.append('\n'.join(current_block))
                current_block = [line]
            else:
                current_block.append(line)
        
        if current_block:
            walkthrough.append('\n'.join(current_block))
        
        return '\n\n'.join(walkthrough)

    def generate_response(self, 
                         query: str, 
                         matches: List[ResourceMatch], 
                         query_analysis) -> GeneratedResponse:
        """Generate a response using only local resources."""
        try:
            if not matches:
                return GeneratedResponse(
                    content="No relevant local resources found.",
                    source_type='local',
                    resources_used=[],
                    confidence_score=0.0
                )

            # Sort matches by score
            matches.sort(key=lambda x: x.score, reverse=True)
            best_matches = matches[:3]  # Use top 3 matches for response generation
            
            # Build response sections
            introduction = f"Based on our {query_analysis.year_level} course materials, here's what I found about your question:"
            
            # Main explanation
            explanation = self._generate_explanation(best_matches, query_analysis.query_type)
            
            # Code examples section
            code_section = ""
            if query_analysis.query_type in ['code', 'both']:
                code_section = "\n\nHere are the relevant code examples:\n```\n"
                code_section += self._combine_code_snippets(best_matches)
                code_section += "\n```"
            
            # Detailed walkthrough of the best match
            walkthrough = f"\n\nLet's break down the main implementation:\n{self._format_code_walkthrough(best_matches[0])}"
            
            # Additional context
            context = ""
            if query_analysis.frameworks:
                context += f"\n\nThis implementation uses {', '.join(query_analysis.frameworks)} "
                context += f"which is appropriate for {query_analysis.year_level} level."
            
            # Summary
            summary = "\n\nKey Points to Remember:"
            for i, match in enumerate(best_matches, 1):
                summary += f"\n{i}. {match.purpose}"
            
            # Combine all sections
            full_response = f"{introduction}\n\n{explanation}{code_section}{walkthrough}{context}{summary}"
            
            return GeneratedResponse(
                content=full_response,
                source_type='local',
                resources_used=[match.source_file for match in best_matches],
                confidence_score=max(match.score for match in matches)
            )

        except Exception as e:
            print(f"Error generating local response: {e}")
            return GeneratedResponse(
                content=f"Error generating response from local resources: {str(e)}",
                source_type='error',
                resources_used=[],
                confidence_score=0.0
            )

class APIResponseGenerator:
    """Handles response generation using external API when local resources are insufficient."""
    
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
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

    def generate_response(self, query: str, query_analysis) -> GeneratedResponse:
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
            response += "Sources: " + ", ".join(generated_response.resources_used) + "\n"
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