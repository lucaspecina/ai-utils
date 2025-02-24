"""
Multi-Agent Workflow using Ollama and LangChain

This script implements a multi-agent system where different specialized agents collaborate
to solve complex tasks. The agents communicate with each other and coordinate their efforts
through a central orchestrator.

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
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import Tool
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

# Configure models - these can be changed via command line arguments
RESEARCH_MODEL = "llama3"  # For research and information gathering
REASONING_MODEL = "llama3"  # For logical reasoning and planning
WRITING_MODEL = "llama3"    # For content generation and summarization

class Agent:
    """
    Base class for specialized agents
    
    Each agent has:
    - A name for identification
    - An LLM model for processing
    - A system prompt that defines its role and behavior
    - Optional tools for external capabilities
    - Memory to store conversation history
    """
    
    def __init__(self, name: str, model_name: str, system_prompt: str, tools: List[Tool] = None):
        self.name = name
        # Initialize the Ollama model with the specified model name
        self.model = Ollama(model=model_name)
        self.system_prompt = system_prompt
        self.tools = tools or []
        # Memory to store conversation history
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Create prompt template with system and user messages
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create agent with tools if provided, otherwise use direct LLM calls
        if self.tools:
            self.agent = create_openai_tools_agent(self.model, self.tools, prompt)
            self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        else:
            self.agent = None
            self.agent_executor = None
    
    def run(self, query: str) -> str:
        """
        Process a query and return a response
        
        If the agent has tools, it will use the agent executor
        Otherwise, it will directly call the LLM
        """
        if self.agent_executor:
            # Use the agent executor with tools
            response = self.agent_executor.invoke({"input": query})
            return response["output"]
        else:
            # Direct LLM call without tools
            messages = [HumanMessage(content=query)]
            response = self.model.invoke(messages)
            return response

class ResearchAgent(Agent):
    """
    Agent specialized in research and information gathering
    
    This agent uses Wikipedia and web search tools to find
    relevant information for the user's query.
    """
    
    def __init__(self, model_name=RESEARCH_MODEL):
        # Set up research tools
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        search = DuckDuckGoSearchRun()
        
        tools = [
            Tool(
                name="Wikipedia",
                func=wikipedia.run,
                description="Useful for searching for information on Wikipedia"
            ),
            Tool(
                name="Web Search",
                func=search.run,
                description="Useful for searching the web for current information"
            )
        ]
        
        system_prompt = """You are a Research Agent specialized in gathering accurate information.
        Your goal is to find relevant, factual information to answer queries.
        Use the tools available to search for information. Be thorough and cite your sources."""
        
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
    parser = argparse.ArgumentParser(description='Multi-Agent System using Ollama and LangChain')
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
        print("Multi-Agent System (Ollama + LangChain)")
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