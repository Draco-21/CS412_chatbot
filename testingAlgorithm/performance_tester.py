import time
import json
import statistics
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid GUI issues
import matplotlib.pyplot as plt
from datetime import datetime
import os
import warnings
import signal
import sys
from chatbot_trial import call_external_api, get_api_key
from retrievalAlgorithms.keyword_retriever import search_keyword_index, create_keyword_index, KEYWORD_INDEX_DIR
import streamlit as st
import re

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="CropBox missing")

def cleanup_resources():
    """Clean up resources before exit."""
    try:
        plt.close('all')
        # Clean up any temporary index files
        if os.path.exists(os.path.join(KEYWORD_INDEX_DIR, "MAIN.tmp")):
            try:
                os.remove(os.path.join(KEYWORD_INDEX_DIR, "MAIN.tmp"))
            except Exception as e:
                print(f"Warning: Could not remove temporary index file: {e}")
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

def setup_signal_handlers():
    """Set up signal handlers if in main thread."""
    try:
        def signal_handler(signum, frame):
            print("\nCleaning up resources...")
            cleanup_resources()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except ValueError:
        # Ignore if not in main thread
        pass

# Initialize Streamlit state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    if 'GOOGLE_API_KEY' in os.environ:
        st.session_state.GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    st.session_state.index_built = False
    st.session_state.show_test_button = False

def check_index_exists():
    """Check if keyword index exists and is valid."""
    try:
        # Check if directory exists and is writable
        if not os.path.exists(KEYWORD_INDEX_DIR):
            os.makedirs(KEYWORD_INDEX_DIR, exist_ok=True)
        
        # Test write permissions
        test_file = os.path.join(KEYWORD_INDEX_DIR, "test_write")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            print(f"Warning: Directory not writable: {e}")
            return False
            
        # Check for index files
        return os.path.exists(os.path.join(KEYWORD_INDEX_DIR, "_MAIN_LOCK"))
    except Exception as e:
        print(f"Error checking index: {e}")
        return False

def check_setup():
    """Check if all necessary components are set up."""
    # Check API Key
    api_key = get_api_key()
    if not api_key:
        st.error("🔑 Google API Key not set! Please provide it in the sidebar.")
        return False
        
    # Check for keyword index and directory permissions
    try:
        if not check_index_exists():
            st.error("📚 Keyword Index not found or directory not writable!")
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                if st.button("Build Keyword Index", key="build_index_button"):
                    with st.spinner("Building keyword index... This may take a few minutes..."):
                        try:
                            # Clean up any existing temporary files first
                            cleanup_resources()
                            success, duration = create_keyword_index()
                            if success:
                                st.session_state.index_built = True
                                st.session_state.show_test_button = True
                                st.success(f"✅ Keyword Index built successfully in {duration:.1f} seconds!")
                            else:
                                st.error("❌ Failed to build keyword index. Check console for errors.")
                                return False
                        except Exception as e:
                            st.error(f"❌ Error building index: {e}")
                            return False
                else:
                    st.info("Click the button above to build the keyword index before running tests.")
                    return False
        else:
            st.session_state.index_built = True
            st.session_state.show_test_button = True
    except Exception as e:
        st.error(f"Error during setup: {e}")
        return False
    
    return True

class ChatbotPerformanceTester:
    def __init__(self):
        self.results_dir = "performance_results"
        self.metrics = {
            "response_time": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "error_count": 0,
            "code_snippet_count": 0,
            "context_relevance": [],
            "response_length": [],
            "keyword_retrieval_time": [],
            "keyword_hits": 0,
            "retrieval_success_rate": []
        }
        self.test_cases = self._load_test_cases()
        
        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def _load_test_cases(self) -> List[Dict]:
        """Load test cases for different academic levels."""
        return [
            {
                "year_level": "Year 1 Certificate",
                "questions": [
                    "Explain how to implement a bubble sort in C++",
                    "What is object-oriented programming?",
                    "How do I use pointers in C++?",
                    "Explain file I/O operations",
                    "What are data structures?"
                ]
            },
            {
                "year_level": "Year 2 Diploma",
                "questions": [
                    "How to create a REST API using Spring Boot?",
                    "Explain multithreading in Java",
                    "What is the difference between SQL and NoSQL?",
                    "How to implement a binary search tree?",
                    "Explain Android activity lifecycle"
                ]
            },
            {
                "year_level": "Year 3 Degree",
                "questions": [
                    "Explain TCP/IP protocol stack",
                    "How to implement a distributed system?",
                    "What is the CAP theorem?",
                    "Explain load balancing algorithms",
                    "How to implement microservices architecture?"
                ]
            },
            {
                "year_level": "Year 4 Postgraduate Diploma",
                "questions": [
                    "Explain backpropagation in neural networks",
                    "How to implement gradient descent?",
                    "What is the difference between supervised and unsupervised learning?",
                    "Explain clustering algorithms",
                    "How to implement a perceptron?"
                ]
            }
        ]

    def _test_single_query(self, question: str, year_level: str):
        """Test a single query and collect metrics."""
        try:
            # Test Keyword Search RAG
            start_time = time.time()
            keyword_content = search_keyword_index(question, year_level)
            keyword_time = time.time() - start_time
            self.metrics["keyword_retrieval_time"].append(keyword_time)
            
            if keyword_content:
                self.metrics["keyword_hits"] += 1
                augmented_prompt = f"""You are a helpful assistant for a {year_level} student.
                Based *primarily* on the following retrieved context from local documents, synthesize a comprehensive and helpful answer to the student's question.
                If the context is insufficient or doesn't directly answer, you may use your general knowledge but clearly state that the provided context was limited.

                Retrieved Context:
                ---
                {keyword_content[:3500]} 
                ---
                Student's Question: "{question}"

                Synthesized Answer:
                """
                self.metrics["retrieval_success_rate"].append(1)
            else:
                augmented_prompt = question
                self.metrics["retrieval_success_rate"].append(0)
            
            # Measure response time for the complete process
            start_time = time.time()
            response = call_external_api(augmented_prompt, [], year_level)
            response_time = time.time() - start_time
            
            # Update metrics
            self.metrics["response_time"].append(response_time)
            self.metrics["response_length"].append(len(response))
            
            # Count code snippets
            code_snippets = self._count_code_snippets(response)
            self.metrics["code_snippet_count"] += code_snippets
            
            # Calculate context relevance
            relevance_score = self._calculate_context_relevance(question, response)
            self.metrics["context_relevance"].append(relevance_score)
            
        except Exception as e:
            print(f"Error testing query: {question}")
            print(f"Error: {e}")
            self.metrics["error_count"] += 1

    def _count_code_snippets(self, text: str) -> int:
        """Count the number of code snippets in the response."""
        code_pattern = r"```(\w+)?\n(.*?)\n```"
        return len(re.findall(code_pattern, text, re.DOTALL))

    def _calculate_context_relevance(self, question: str, response: str) -> float:
        """Calculate the relevance score of the response to the question."""
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        common_words = question_words.intersection(response_words)
        return len(common_words) / len(question_words) if question_words else 0

    def run_performance_test(self) -> Dict:
        """Run comprehensive performance tests."""
        print("Starting performance tests...")
        start_time = time.time()
        
        for test_case in self.test_cases:
            year_level = test_case["year_level"]
            print(f"\nTesting {year_level}...")
            
            for question in test_case["questions"]:
                self._test_single_query(question, year_level)
        
        total_time = time.time() - start_time
        return self._generate_report(total_time)

    def _generate_report(self, total_time: float) -> Dict:
        """Generate a comprehensive performance report."""
        # Helper function to safely calculate mean
        def safe_mean(data_list):
            return statistics.mean(data_list) if data_list else 0.0

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_test_time": total_time,
            # Response time metrics
            "average_response_time": safe_mean(self.metrics["response_time"]),
            "median_response_time": statistics.median(self.metrics["response_time"]) if self.metrics["response_time"] else 0.0,
            "min_response_time": min(self.metrics["response_time"]) if self.metrics["response_time"] else 0.0,
            "max_response_time": max(self.metrics["response_time"]) if self.metrics["response_time"] else 0.0,
            # Retrieval metrics
            "average_keyword_retrieval_time": safe_mean(self.metrics["keyword_retrieval_time"]),
            "keyword_hit_rate": self.metrics["keyword_hits"] / len(self.metrics["keyword_retrieval_time"]) if self.metrics["keyword_retrieval_time"] else 0,
            "retrieval_success_rate": safe_mean(self.metrics["retrieval_success_rate"]),
            # Other metrics
            "cache_hit_rate": self.metrics["cache_hits"] / (self.metrics["cache_hits"] + self.metrics["cache_misses"]) if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0 else 0,
            "error_rate": self.metrics["error_count"] / len(self.metrics["response_time"]) if self.metrics["response_time"] else 0,
            "average_code_snippets_per_response": self.metrics["code_snippet_count"] / len(self.metrics["response_time"]) if self.metrics["response_time"] else 0,
            "average_context_relevance": safe_mean(self.metrics["context_relevance"]),
            "average_response_length": safe_mean(self.metrics["response_length"])
        }
        
        # Save report to file
        report_file = os.path.join(self.results_dir, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        return report

    def visualize_results(self):
        """Create visualizations of the performance metrics."""
        try:
            # Clear any existing plots
            plt.close('all')
            
            # Create figure with tight layout
            fig = plt.figure(figsize=(15, 15), constrained_layout=True)
            
            # Create subplot grid
            gs = fig.add_gridspec(3, 2)
            
            # Response Time Distribution
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.hist(self.metrics["response_time"], bins=20)
            ax1.set_title("Response Time Distribution")
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Frequency")
            
            # Keyword Retrieval Time Distribution
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.hist(self.metrics["keyword_retrieval_time"], bins=20)
            ax2.set_title("Keyword Retrieval Time Distribution")
            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("Frequency")
            
            # Context Relevance vs Response Time
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.scatter(self.metrics["response_time"], self.metrics["context_relevance"])
            ax3.set_title("Context Relevance vs Response Time")
            ax3.set_xlabel("Response Time (seconds)")
            ax3.set_ylabel("Context Relevance Score")
            
            # Retrieval Success Rate
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.hist(self.metrics["retrieval_success_rate"], bins=2)
            ax4.set_title("Retrieval Success Distribution")
            ax4.set_xlabel("Success (0=No Retrieval, 1=Retrieved)")
            ax4.set_ylabel("Count")
            
            # Response Length Distribution
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.hist(self.metrics["response_length"], bins=20)
            ax5.set_title("Response Length Distribution")
            ax5.set_xlabel("Length (characters)")
            ax5.set_ylabel("Frequency")
            
            # Code Snippets Count
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.bar(["With Code", "Without Code"], 
                    [self.metrics["code_snippet_count"], 
                     len(self.metrics["response_time"]) - self.metrics["code_snippet_count"]])
            ax6.set_title("Responses with/without Code Snippets")
            
            # Save the plot with high quality settings
            plot_file = os.path.join(self.results_dir, f"performance_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', format='png')
            plt.close('all')  # Clean up
            
            return plot_file
        except Exception as e:
            print(f"Error in visualization: {e}")
            return None

def run_performance_test():
    """Run the performance test and display results in Streamlit."""
    st.title("Chatbot Performance Test Results")
    
    # Check setup before proceeding
    if not check_setup():
        return
    
    # Show the Run Performance Test button if index exists
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.session_state.show_test_button:
            if st.button("Run Performance Test", key="run_test_button", use_container_width=True):
                with st.spinner("Running performance tests..."):
                    try:
                        tester = ChatbotPerformanceTester()
                        report = tester.run_performance_test()
                        plot_file = tester.visualize_results()
                        
                        # Display results
                        st.subheader("Performance Metrics")
                        
                        # Response Time Metrics
                        st.write("Response Time Metrics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Response Time", f"{report['average_response_time']:.2f}s")
                            st.metric("Min Response Time", f"{report['min_response_time']:.2f}s")
                        with col2:
                            st.metric("Median Response Time", f"{report['median_response_time']:.2f}s")
                            st.metric("Max Response Time", f"{report['max_response_time']:.2f}s")
                        with col3:
                            st.metric("Error Rate", f"{report['error_rate']*100:.1f}%")
                        
                        # Retrieval Metrics
                        st.write("Retrieval Metrics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Keyword Search Time", f"{report['average_keyword_retrieval_time']:.2f}s")
                            st.metric("Keyword Hit Rate", f"{report['keyword_hit_rate']*100:.1f}%")
                        with col2:
                            st.metric("Overall Retrieval Success", f"{report['retrieval_success_rate']*100:.1f}%")
                        
                        # Content Quality Metrics
                        st.write("Content Quality Metrics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Average Code Snippets", f"{report['average_code_snippets_per_response']:.1f}")
                            st.metric("Context Relevance", f"{report['average_context_relevance']*100:.1f}%")
                        with col2:
                            st.metric("Average Response Length", f"{report['average_response_length']:.0f} chars")
                        
                        # Display plots if available
                        if plot_file and os.path.exists(plot_file):
                            st.subheader("Performance Visualizations")
                            st.image(plot_file)
                        
                        # Display detailed report
                        st.subheader("Detailed Report")
                        st.json(report)
                        
                    except Exception as e:
                        st.error(f"Error during performance testing: {str(e)}")
                        print(f"Error details: {e}")

if __name__ == "__main__":
    try:
        # Set up signal handlers in main thread
        setup_signal_handlers()
        run_performance_test()
    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
        cleanup_resources()
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        cleanup_resources()
        sys.exit(1) 