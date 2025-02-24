"""
Agentic memory management system using LangGraph for workflow orchestration.

This implementation creates a graph of specialized agents that work together to:
1. Analyze user input
2. Retrieve relevant memories
3. Generate responses with context
4. Create and store new memories

Usage:
    python langgraph_memory.py --model-name deepseek-r1:8b
"""

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from typing import Dict, List, TypedDict, Annotated, Literal
import chromadb
from chromadb import Client, Settings
from datetime import datetime
import json
import pprint

# Define the state schema for our graph
class AgentState(TypedDict):
    user_input: str
    retrieved_memories: Dict[str, List[str]]
    analysis: Dict[str, str]
    response: str
    new_memories: Dict[str, str]
    debug_info: Dict[str, str]

class MemoryAgent:
    def __init__(self, model_name="deepseek-r1:8b"):
        print("\n=== INITIALIZING AGENTIC MEMORY SYSTEM ===")
        print("- Creating Ollama chat model")
        self.model = ChatOllama(model=model_name)
        
        print("- Initializing ChromaDB")
        self.db = Client(Settings(allow_reset=True))
        
        print("- Creating memory collections")
        self.user_memories = self.db.create_collection("user_memories")
        self.conversation_memories = self.db.create_collection("conversation_memories")
        self.fact_memories = self.db.create_collection("fact_memories")
        
        print("- Building agent workflow graph")
        self.workflow = self._build_workflow()
        print("=== AGENTIC MEMORY SYSTEM READY ===\n")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for memory operations"""
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes to the graph
        workflow.add_node("analyze_input", self._analyze_input)
        workflow.add_node("retrieve_memories", self._retrieve_memories)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("create_memories", self._create_memories)
        workflow.add_node("store_memories", self._store_memories)
        
        # Define the edges (workflow)
        workflow.add_edge("analyze_input", "retrieve_memories")
        workflow.add_edge("retrieve_memories", "generate_response")
        workflow.add_edge("generate_response", "create_memories")
        workflow.add_edge("create_memories", "store_memories")
        workflow.add_edge("store_memories", END)
        
        # Set the entry point
        workflow.set_entry_point("analyze_input")
        
        # Compile the graph
        return workflow.compile()
    
    def _analyze_input(self, state: AgentState) -> AgentState:
        """Analyze user input to determine intent and key concepts"""
        print("\n" + "="*50)
        print("STEP 1: ANALYZING USER INPUT")
        print("="*50)
        
        prompt = ChatPromptTemplate.from_template("""
        Analyze the following user input to identify:
        1. The main intent/purpose
        2. Key topics or concepts mentioned
        3. Any emotional tone or sentiment
        4. Specific information requests
        
        User input: {user_input}
        
        Respond with a JSON object containing these keys:
        - intent: the main purpose of the message
        - topics: list of key topics
        - sentiment: emotional tone
        - info_requests: any specific information being requested
        """)
        
        chain = prompt | self.model | StrOutputParser()
        
        try:
            result = chain.invoke({"user_input": state["user_input"]})
            analysis = json.loads(result)
        except:
            analysis = {
                "intent": "unknown",
                "topics": ["general"],
                "sentiment": "neutral",
                "info_requests": []
            }
        
        print("\nINPUT ANALYSIS:")
        print("-"*30)
        pprint.pprint(analysis)
        
        return {
            **state,
            "analysis": analysis,
            "debug_info": {**state.get("debug_info", {}), "analysis": result}
        }
    
    def _retrieve_memories(self, state: AgentState) -> AgentState:
        """Retrieve relevant memories based on analyzed input"""
        print("\n" + "="*50)
        print("STEP 2: RETRIEVING RELEVANT MEMORIES")
        print("="*50)
        
        user_input = state["user_input"]
        analysis = state["analysis"]
        
        # Create a more targeted query using the analysis
        topics = " ".join(analysis.get("topics", []))
        intent = analysis.get("intent", "")
        query = f"{user_input} {topics} {intent}"
        
        print(f"Searching memories with enhanced query: '{query}'")
        
        # Query each memory collection
        relevant_memories = {
            "User Information": self.user_memories.query(
                query_texts=[query],
                n_results=3
            ).get('documents', [[]])[0],
            
            "Conversation History": self.conversation_memories.query(
                query_texts=[query],
                n_results=3
            ).get('documents', [[]])[0],
            
            "Facts & Knowledge": self.fact_memories.query(
                query_texts=[query],
                n_results=3
            ).get('documents', [[]])[0]
        }
        
        print("\nRELEVANT MEMORIES FOUND:")
        print("-"*30)
        for mem_type, memories in relevant_memories.items():
            print(f"\n📎 {mem_type}:")
            if memories:
                for mem in memories:
                    print(f"  • {mem}")
            else:
                print("  • No memories yet")
        
        return {
            **state,
            "retrieved_memories": relevant_memories
        }
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate response using retrieved memories and analysis"""
        print("\n" + "="*50)
        print("STEP 3: GENERATING RESPONSE")
        print("="*50)
        
        # Format memories for context
        context_parts = []
        for memory_type, memory_list in state["retrieved_memories"].items():
            if memory_list:
                context_parts.append(f"{memory_type}:")
                context_parts.extend([f"- {mem}" for mem in memory_list])
        
        memory_context = "\n\n".join(context_parts) if context_parts else "No relevant memories found."
        
        # Create a prompt that uses both memories and analysis
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant with access to memory.
        
        MEMORY CONTEXT:
        {memory_context}
        
        USER INTENT ANALYSIS:
        - Main intent: {analysis[intent]}
        - Key topics: {analysis[topics]}
        - Emotional tone: {analysis[sentiment]}
        
        USER MESSAGE:
        {user_input}
        
        Respond naturally to the user's message, using the memories to maintain conversation continuity.
        Be helpful, informative, and personalized based on what you know about the user.
        """)
        
        chain = prompt | self.model | StrOutputParser()
        
        response = chain.invoke({
            "memory_context": memory_context,
            "analysis": state["analysis"],
            "user_input": state["user_input"]
        })
        
        print("\nGENERATED RESPONSE:")
        print("-"*30)
        print(response)
        
        return {
            **state,
            "response": response
        }
    
    def _create_memories(self, state: AgentState) -> AgentState:
        """Create new memories from the conversation"""
        print("\n" + "="*50)
        print("STEP 4: CREATING NEW MEMORIES")
        print("="*50)
        
        prompt = ChatPromptTemplate.from_template("""
        Analyze this exchange and create three types of memories:
        
        Exchange:
        User: {user_input}
        Assistant: {response}
        
        User intent analysis: {analysis}
        
        Create three different types of memories:
        1. User Memory: Information about the user's background, preferences, and traits
        2. Conversation Memory: The flow, context, and development of the discussion
        3. Fact Memory: Specific facts, information, or knowledge shared
        
        Format as JSON with these keys: "user", "conversation", "facts"
        Each value should be a detailed string describing the memory.
        Only include memories if there's meaningful information to store.
        """)
        
        chain = prompt | self.model | StrOutputParser()
        
        try:
            result = chain.invoke({
                "user_input": state["user_input"],
                "response": state["response"],
                "analysis": state["analysis"]
            })
            new_memories = json.loads(result)
        except:
            new_memories = {
                "user": "",
                "conversation": f"User asked: {state['user_input']}. Assistant responded about {state['analysis'].get('topics', ['general topics'])}.",
                "facts": ""
            }
        
        print("\nNEW MEMORIES CREATED:")
        print("-"*30)
        pprint.pprint(new_memories)
        
        return {
            **state,
            "new_memories": new_memories,
            "debug_info": {**state.get("debug_info", {}), "memory_creation": result}
        }
    
    def _store_memories(self, state: AgentState) -> AgentState:
        """Store the new memories in the vector database"""
        print("\n" + "="*50)
        print("STEP 5: STORING MEMORIES")
        print("="*50)
        
        new_memories = state["new_memories"]
        timestamp = datetime.now().timestamp()
        
        # Store user memories
        if new_memories.get("user"):
            print("\n📌 STORING USER MEMORY:")
            print("-"*30)
            print(new_memories["user"])
            self.user_memories.add(
                documents=[new_memories["user"]],
                ids=[f"user_mem_{timestamp}"],
                metadatas=[{"type": "user", "timestamp": str(datetime.now())}]
            )
        
        # Store conversation memories
        if new_memories.get("conversation"):
            print("\n📌 STORING CONVERSATION MEMORY:")
            print("-"*30)
            print(new_memories["conversation"])
            self.conversation_memories.add(
                documents=[new_memories["conversation"]],
                ids=[f"conv_mem_{timestamp}"],
                metadatas=[{"type": "conversation", "timestamp": str(datetime.now())}]
            )
        
        # Store fact memories
        if new_memories.get("facts"):
            print("\n📌 STORING FACT MEMORY:")
            print("-"*30)
            print(new_memories["facts"])
            self.fact_memories.add(
                documents=[new_memories["facts"]],
                ids=[f"fact_mem_{timestamp}"],
                metadatas=[{"type": "fact", "timestamp": str(datetime.now())}]
            )
        
        print("\n=== MEMORY STORAGE COMPLETE ===")
        
        return state
    
    def chat(self, user_input: str) -> str:
        """Process a user message through the agent workflow"""
        print("\n" + "="*70)
        print("STARTING AGENTIC MEMORY WORKFLOW")
        print("="*70)
        
        # Initialize the state with user input
        initial_state = AgentState(
            user_input=user_input,
            retrieved_memories={},
            analysis={},
            response="",
            new_memories={},
            debug_info={}
        )
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        print("\n" + "="*70)
        print("WORKFLOW COMPLETED")
        print("="*70)
        
        return final_state["response"]
    
    def clear(self):
        """Clear all memories"""
        print("\n=== CLEARING ALL MEMORIES ===")
        self.db.reset()
        # Recreate collections
        self.user_memories = self.db.create_collection("user_memories")
        self.conversation_memories = self.db.create_collection("conversation_memories")
        self.fact_memories = self.db.create_collection("fact_memories")
        print("✓ All memories cleared")
        print("=== MEMORY SYSTEM RESET ===")

class MemoryManagerCLI:
    def __init__(self, model_name="deepseek-r1:8b"):
        self.memory = MemoryAgent(model_name=model_name)

    def run(self):
        """Main CLI loop"""
        print("\nLangGraph Agentic Memory Manager")
        print("Enter messages or commands (/help for list)")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue
                
                if user_input:
                    response = self.memory.chat(user_input)
                    print(f"\nAssistant: {response}")
                    
            except KeyboardInterrupt:
                print("\nUse /quit to exit")
                continue
            except Exception as e:
                print(f"\nError: {str(e)}")

    def _handle_command(self, command):
        """Handle CLI commands"""
        cmd = command[1:].lower().split()
        
        if cmd[0] == 'quit':
            print("Goodbye!")
            exit()
        elif cmd[0] == 'clear':
            self.memory.clear()
        elif cmd[0] == 'help':
            self._print_help()
        elif cmd[0] == 'debug':
            print("Debug mode not implemented yet")
        else:
            print(f"Unknown command: {command}")

    def _print_help(self):
        """Show available commands"""
        print("\nCommands:")
        print("  /quit   - Exit the program")
        print("  /clear  - Clear all memories")
        print("  /debug  - Toggle debug mode")
        print("  /help   - Show this help message")

def main():
    """Start the CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LangGraph Agentic Memory Manager')
    parser.add_argument('--model-name', default='deepseek-r1:8b', help='Name of the Ollama model to use')
    
    args = parser.parse_args()
    
    cli = MemoryManagerCLI(model_name=args.model_name)
    cli.run()

if __name__ == "__main__":
    main() 