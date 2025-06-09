import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load environment variables
load_dotenv()

# Add the current directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
chatbot_dir = os.path.join(project_root, "chatbotcodes")
sys.path.extend([project_root, chatbot_dir])

# Import core chatbot functionality
from retrievalAlgorithms.resource_retriever import SmartResourceRetriever
from chatbotcodes.query_analyzer import QueryAnalyzer
from chatbotcodes.rag_generator import LocalResponseGenerator, APIResponseGenerator

class ResponseQualityMetrics:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        
    def calculate_completeness_score(self, response):
        """Calculate completeness score based on response length and structure"""
        words = word_tokenize(response)
        score = min(len(words) / 100.0, 1.0)  # Normalize to 0-1
        
        # Check for key components
        has_explanation = any(marker in response.lower() for marker in ["because", "therefore", "this means", "explains"])
        has_code = "```" in response or any(marker in response.lower() for marker in ["code", "function", "class", "def "])
        has_example = any(marker in response.lower() for marker in ["example", "instance", "case"])
        
        component_score = (has_explanation + has_code + has_example) / 3.0
        return (score + component_score) / 2.0

    def calculate_relevance_score(self, query, response):
        """Calculate relevance using cosine similarity"""
        try:
            tfidf_matrix = self.vectorizer.fit_transform([query, response])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            return 0.0

    def calculate_code_quality_score(self, response):
        """Evaluate the quality of code examples if present"""
        if "```" not in response:
            return 0.0
            
        code_blocks = response.split("```")[1::2]  # Get content between ``` markers
        if not code_blocks:
            return 0.0
            
        scores = []
        for code in code_blocks:
            # Check for code quality indicators
            has_comments = "#" in code or "//" in code
            has_proper_indentation = any(line.startswith(("    ", "\t")) for line in code.split("\n"))
            has_functions = any(marker in code for marker in ["def ", "function", "class"])
            
            block_score = (has_comments + has_proper_indentation + has_functions) / 3.0
            scores.append(block_score)
            
        return sum(scores) / len(scores)

    def evaluate_response(self, query, response):
        """Calculate overall quality score"""
        completeness = self.calculate_completeness_score(response)
        relevance = self.calculate_relevance_score(query, response)
        code_quality = self.calculate_code_quality_score(response)
        
        # Weighted average of scores
        overall_score = (0.4 * completeness + 0.4 * relevance + 0.2 * code_quality)
        
        return {
            'completeness': completeness,
            'relevance': relevance,
            'code_quality': code_quality,
            'overall': overall_score
        }

def get_response(query, is_trial=True):
    """Core response generation without Streamlit dependencies"""
    retriever = SmartResourceRetriever()
    analyzer = QueryAnalyzer()
    local_generator = LocalResponseGenerator()
    api_generator = APIResponseGenerator()
    
    # Process query
    year_level = "Year 3 Degree"  # Default to Year 3
    matches = retriever.search_resources(query, year_level)
    analysis = analyzer.analyze_query(query, year_level)
    
    if is_trial:
        response = local_generator.generate_response(query, matches, analysis)
    else:
        response = api_generator.generate_response(query, analysis)
    
    return response.content

def plot_results(trial_times, api_times, trial_scores, api_scores, test_cases):
    """Plot both time and quality metrics"""
    fig = plt.figure(figsize=(15, 10))
    
    # Time comparison
    ax1 = plt.subplot(2, 2, 1)
    x = np.arange(len(test_cases))
    width = 0.35
    
    ax1.bar(x - width/2, trial_times, width, label='Trial Chatbot')
    ax1.bar(x + width/2, api_times, width, label='API Chatbot')
    ax1.set_ylabel('Response Time (seconds)')
    ax1.set_title('Response Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Test {i+1}' for i in range(len(test_cases))], rotation=45)
    ax1.legend()

    # Average response time
    ax2 = plt.subplot(2, 2, 2)
    avg_times = [np.mean(trial_times), np.mean(api_times)]
    ax2.bar(['Trial Chatbot', 'API Chatbot'], avg_times)
    ax2.set_ylabel('Average Response Time (seconds)')
    ax2.set_title('Average Response Time')

    # Quality metrics comparison
    ax3 = plt.subplot(2, 2, 3)
    metrics = ['Completeness', 'Relevance', 'Code Quality', 'Overall']
    trial_means = [
        np.mean([s['completeness'] for s in trial_scores]),
        np.mean([s['relevance'] for s in trial_scores]),
        np.mean([s['code_quality'] for s in trial_scores]),
        np.mean([s['overall'] for s in trial_scores])
    ]
    api_means = [
        np.mean([s['completeness'] for s in api_scores]),
        np.mean([s['relevance'] for s in api_scores]),
        np.mean([s['code_quality'] for s in api_scores]),
        np.mean([s['overall'] for s in api_scores])
    ]
    
    x = np.arange(len(metrics))
    ax3.bar(x - width/2, trial_means, width, label='Trial Chatbot')
    ax3.bar(x + width/2, api_means, width, label='API Chatbot')
    ax3.set_ylabel('Score')
    ax3.set_title('Quality Metrics Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, rotation=45)
    ax3.legend()

    # Overall quality comparison over time
    ax4 = plt.subplot(2, 2, 4)
    trial_overall = [s['overall'] for s in trial_scores]
    api_overall = [s['overall'] for s in api_scores]
    ax4.plot(range(1, len(test_cases) + 1), trial_overall, 'b-', label='Trial Chatbot')
    ax4.plot(range(1, len(test_cases) + 1), api_overall, 'r-', label='API Chatbot')
    ax4.set_xlabel('Test Case')
    ax4.set_ylabel('Overall Quality Score')
    ax4.set_title('Quality Score Progression')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('chatbot_comparison_results.png')
    plt.close()

def main():
    # Define test cases
    test_cases = [
        "What are the course prerequisites?",
        "Tell me about the grading policy",
        "When are office hours?",
        "What is the late submission policy?",
        "How do I submit assignments?",
        "What programming languages will we use?",
        "Is attendance mandatory?",
        "What is the exam format?",
        "Are there any group projects?",
        "What is the course schedule?"
    ]

    trial_times = []
    api_times = []
    trial_scores = []
    api_scores = []
    quality_metrics = ResponseQualityMetrics()

    print("\nRunning tests...\n")
    print("=" * 50)

    # Run tests
    for i, query in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: '{query}'")
        
        # Test Trial Chatbot
        start_time = time.time()
        trial_response = get_response(query, True)
        trial_time = time.time() - start_time
        trial_times.append(trial_time)
        
        # Evaluate trial response
        trial_quality = quality_metrics.evaluate_response(query, trial_response)
        trial_scores.append(trial_quality)
        
        print(f"\nTrial Chatbot Response Time: {trial_time:.2f} seconds")
        print(f"Trial Chatbot Quality Scores:")
        print(f"- Completeness: {trial_quality['completeness']:.2f}")
        print(f"- Relevance: {trial_quality['relevance']:.2f}")
        print(f"- Code Quality: {trial_quality['code_quality']:.2f}")
        print(f"- Overall: {trial_quality['overall']:.2f}")
        print(f"Response Preview: {trial_response[:100]}...")

        # Test API Chatbot
        start_time = time.time()
        api_response = get_response(query, False)
        api_time = time.time() - start_time
        api_times.append(api_time)
        
        # Evaluate API response
        api_quality = quality_metrics.evaluate_response(query, api_response)
        api_scores.append(api_quality)
        
        print(f"\nAPI Chatbot Response Time: {api_time:.2f} seconds")
        print(f"API Chatbot Quality Scores:")
        print(f"- Completeness: {api_quality['completeness']:.2f}")
        print(f"- Relevance: {api_quality['relevance']:.2f}")
        print(f"- Code Quality: {api_quality['code_quality']:.2f}")
        print(f"- Overall: {api_quality['overall']:.2f}")
        print(f"Response Preview: {api_response[:100]}...")
        
        print("\n" + "=" * 50)

    # Plot results
    plot_results(trial_times, api_times, trial_scores, api_scores, test_cases)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nResponse Times:")
    print(f"Trial Chatbot Average: {np.mean(trial_times):.2f} seconds")
    print(f"API Chatbot Average: {np.mean(api_times):.2f} seconds")
    
    print("\nQuality Metrics:")
    print("\nTrial Chatbot Averages:")
    print(f"- Completeness: {np.mean([s['completeness'] for s in trial_scores]):.2f}")
    print(f"- Relevance: {np.mean([s['relevance'] for s in trial_scores]):.2f}")
    print(f"- Code Quality: {np.mean([s['code_quality'] for s in trial_scores]):.2f}")
    print(f"- Overall: {np.mean([s['overall'] for s in trial_scores]):.2f}")
    
    print("\nAPI Chatbot Averages:")
    print(f"- Completeness: {np.mean([s['completeness'] for s in api_scores]):.2f}")
    print(f"- Relevance: {np.mean([s['relevance'] for s in api_scores]):.2f}")
    print(f"- Code Quality: {np.mean([s['code_quality'] for s in api_scores]):.2f}")
    print(f"- Overall: {np.mean([s['overall'] for s in api_scores]):.2f}")
    
    print(f"\nResults have been saved to 'chatbot_comparison_results.png'")

if __name__ == "__main__":
    main() 