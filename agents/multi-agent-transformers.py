"""
Multi-Agent Workflow using Hugging Face Transformers

This script implements a multi-agent system where different specialized agents collaborate
to solve complex tasks using Hugging Face Transformer models.

Pattern:
1. User query is received by the Orchestrator
2. Research Agent gathers information using tools
3. Reasoning Agent analyzes the information
4. Writing Agent produces the final response
5. Result is returned to the user
"""

import os
import argparse
import json
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import wikipedia
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# Configure models - you can replace these with your preferred models
RESEARCH_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # For research
REASONING_MODEL = "google/flan-t5-large"          # For reasoning
WRITING_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # For writing

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

class Tool:
    """
    Simple tool implementation
    
    Each tool has:
    - A name for identification
    - A function to execute
    - A description of its capabilities
    """
    
    def __init__(self, name: str, func: callable, description: str):
        self.name = name
        self.func = func
        self.description = description
    
    def run(self, query: str) -> str:
        """Execute the tool's function with the given query"""
        return self.func(query)

class Agent:
    """
    Base class for specialized agents
    
    Each agent has:
    - A name for identification
    - A model and tokenizer for processing
    - A system prompt that defines its role and behavior
    - Optional tools for external capabilities
    - Conversation history storage
    """
    
    def __init__(self, name: str, model_name: str, system_prompt: str, tools: List[Tool] = None):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.conversation_history = []
        
        # Load model and tokenizer
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use different loading strategies based on model size and available hardware
        try:
            # Try to load the full model with optimizations when possible
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
        except Exception as e:
            # Fall back to pipeline implementation if full model loading fails
            print(f"Error loading full model: {e}")
            print("Falling back to pipeline implementation...")
            self.model = None
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=self.tokenizer,
                max_length=1024,
                device=0 if device == "cuda" else -1
            )
    
    def _format_prompt(self, query: str) -> str:
        """
        Format the prompt based on the model type
        
        This creates a standardized prompt format with system instructions
        and the user query.
        """
        # This is a simplified prompt format - adjust based on your specific models
        return f"{self.system_prompt}\n\nQuery: {query}\n\nResponse:"
    
    def run(self, query: str) -> str:
        """
        Process a query and return a response
        
        First checks if tools should be used, otherwise uses the model.
        """
        # First check if we should use tools
        if self.tools and "use tool" in query.lower():
            for tool in self.tools:
                if tool.name.lower() in query.lower():
                    tool_query = query.split("use tool")[1].strip()
                    return tool.run(tool_query)
        
        # Otherwise use the model
        formatted_prompt = self._format_prompt(query)
        
        if self.model is not None:
            # Direct model inference when full model is loaded
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=1024,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the response part
            response = response.split("Response:")[1].strip() if "Response:" in response else response
        else:
            # Pipeline inference when using the fallback pipeline
            response = self.pipeline(formatted_prompt)[0]["generated_text"]
            response = response.replace(formatted_prompt, "").strip()
        
        # Store the interaction in conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response

def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for information
    
    Args:
        query: The search query
        
    Returns:
        A string containing the Wikipedia search results
    """
    try:
        results = wikipedia.search(query)
        if not results:
            return "No Wikipedia results found."
        
        page = wikipedia.page(results[0])
        summary = page.summary
        return f"Wikipedia ({page.title}): {summary[:1000]}..."
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

def search_web(query: str) -> str:
    """
    Search the web using DuckDuckGo
    
    Args:
        query: The search query
        
    Returns:
        A string containing formatted search results
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        
        if not results:
            return "No search results found."
        
        formatted_results = "\n\n".join([
            f"Title: {r['title']}\nLink: {r['href']}\nContent: {r['body'][:300]}..."
            for r in results
        ])
        
        return formatted_results
    except Exception as e:
        return f"Error searching the web: {str(e)}"

class ResearchAgent(Agent):
    """
    Agent specialized in research and information gathering
    
    This agent uses Wikipedia and web search tools to find
    relevant information for the user's query.
    """
    
    def __init__(self, model_name=RESEARCH_MODEL):
        # Set up research tools
        tools = [
            Tool(
                name="Wikipedia",
                func=search_wikipedia,
                description="Search Wikipedia for information"
            ),
            Tool(
                name="Web Search",
                func=search_web,
                description="Search the web for information"
            )
        ]
        
        system_prompt = """You are a Research Agent specialized in gathering accurate information.
        Your goal is to find relevant, factual information to answer queries.
        Be thorough and cite your sources."""
        
        super().__init__("Research Agent", model_name, system_prompt, tools)

class ReasoningAgent(Agent):
    """
    Agent specialized in logical reasoning and planning
    
    This agent analyzes information, identifies patterns,
    and develops logical plans and insights.
    """
    
    def __init__(self, model_name=REASONING_MODEL):
        system_prompt = """You are a Reasoning Agent specialized in logical analysis and planning.
        Your goal is to analyze information, identify patterns, and develop logical plans.
        Think step by step and explain your reasoning process clearly."""
        
        super().__init__("Reasoning Agent", model_name, system_prompt)

class WritingAgent(Agent):
    """
    Agent specialized in content generation and summarization
    
    This agent creates high-quality, well-structured content
    based on the information and insights provided.
    """
    
    def __init__(self, model_name=WRITING_MODEL):
        system_prompt = """You are a Writing Agent specialized in creating high-quality content.
        Your goal is to generate clear, concise, and engaging text based on the information provided.
        Adapt your writing style to the specific needs of each task."""
        
        super().__init__("Writing Agent", model_name, system_prompt)

class Orchestrator:
    """
    Coordinates the workflow between multiple agents
    
    The orchestrator manages the flow of information between agents:
    1. Research Agent gathers information
    2. Reasoning Agent analyzes the information
    3. Writing Agent produces the final response
    """
    
    def __init__(self, research_model=RESEARCH_MODEL, reasoning_model=REASONING_MODEL, writing_model=WRITING_MODEL):
        # Initialize the specialized agents
        self.research_agent = ResearchAgent(model_name=research_model)
        self.reasoning_agent = ReasoningAgent(model_name=reasoning_model)
        self.writing_agent = WritingAgent(model_name=writing_model)
        self.conversation_history = []
    
    def process_query(self, query: str, verbose: bool = False) -> str:
        """
        Process a complex query using multiple agents
        
        Args:
            query: The user's query
            verbose: Whether to print detailed progress information
            
        Returns:
            The final response from the Writing Agent
        """
        
        # Step 1: Research phase - gather information
        if verbose:
            print("Research Agent working...")
        research_prompt = f"Research the following topic thoroughly: {query}"
        research_results = self.research_agent.run(research_prompt)
        self.conversation_history.append({"agent": "Research", "input": research_prompt, "output": research_results})
        
        # Step 2: Reasoning phase - analyze information and develop a plan
        if verbose:
            print("Reasoning Agent working...")
        reasoning_prompt = f"Analyze this information and develop insights: {research_results}\nOriginal query: {query}"
        reasoning_results = self.reasoning_agent.run(reasoning_prompt)
        self.conversation_history.append({"agent": "Reasoning", "input": reasoning_prompt, "output": reasoning_results})
        
        # Step 3: Writing phase - generate the final response
        if verbose:
            print("Writing Agent working...")
        writing_prompt = f"""
        Create a comprehensive response to the original query: {query}
        
        Research findings:
        {research_results}
        
        Analysis and insights:
        {reasoning_results}
        
        Format your response in a clear, engaging way that directly addresses the original query.
        """
        final_response = self.writing_agent.run(writing_prompt)
        self.conversation_history.append({"agent": "Writing", "input": writing_prompt, "output": final_response})
        
        return final_response
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return the conversation history"""
        return self.conversation_history
    
    def save_conversation(self, filename: str) -> None:
        """Save the conversation history to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)

def main():
    # Set up argument parser for CLI
    parser = argparse.ArgumentParser(description='Multi-Agent System using Hugging Face Transformers')
    parser.add_argument('--research-model', type=str, default=RESEARCH_MODEL, 
                        help=f'Model to use for research (default: {RESEARCH_MODEL})')
    parser.add_argument('--reasoning-model', type=str, default=REASONING_MODEL, 
                        help=f'Model to use for reasoning (default: {REASONING_MODEL})')
    parser.add_argument('--writing-model', type=str, default=WRITING_MODEL, 
                        help=f'Model to use for writing (default: {WRITING_MODEL})')
    parser.add_argument('--query', type=str, help='Query to process (if not provided, interactive mode is used)')
    parser.add_argument('--save', type=str, help='Save conversation history to specified file')
    parser.add_argument('--verbose', action='store_true', help='Show detailed progress information')
    
    args = parser.parse_args()
    
    print("Initializing Multi-Agent System with Transformers...")
    print(f"Using device: {device}")
    
    # Initialize the orchestrator with specified models
    orchestrator = Orchestrator(
        research_model=args.research_model,
        reasoning_model=args.reasoning_model,
        writing_model=args.writing_model
    )
    
    # Process a single query if provided
    if args.query:
        print(f"Processing query: {args.query}")
        if args.verbose:
            print("\nProcessing your query across multiple agents...\n")
        response = orchestrator.process_query(args.query, verbose=args.verbose)
        print("\n=== Final Response ===")
        print(response)
        
        # Save conversation if requested
        if args.save:
            orchestrator.save_conversation(args.save)
            print(f"\nConversation saved to {args.save}")
    else:
        # Interactive mode
        print("\nMulti-Agent System (Hugging Face Transformers)")
        print("Type 'exit' to quit, 'save <filename>' to save conversation")
        print("-" * 50)
        
        while True:
            user_input = input("\nEnter your query: ")
            
            # Check for commands
            if user_input.lower() == 'exit':
                break
            elif user_input.lower().startswith('save '):
                filename = user_input[5:].strip()
                orchestrator.save_conversation(filename)
                print(f"Conversation saved to {filename}")
                continue
            
            # Process the query
            print("\nProcessing your query across multiple agents...\n")
            response = orchestrator.process_query(user_input, verbose=args.verbose)
            
            print("\n=== Final Response ===")
            print(response)
            
            # Optionally show the agent workflow
            print("\nWould you like to see the detailed agent workflow? (y/n)")
            if input().lower() == 'y':
                history = orchestrator.get_conversation_history()
                for i, entry in enumerate(history[-3:]):  # Show only the last interaction
                    print(f"\n--- {entry['agent']} Agent ---")
                    print(f"Output: {entry['output'][:300]}...")  # Show first 300 chars

if __name__ == "__main__":
    main() 