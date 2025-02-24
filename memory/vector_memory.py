"""
Advanced memory management system using ChromaDB for vector storage and RAG for retrieval.
This version stores memories in a structured way and uses semantic search for context.

Usage:
    python vector_memory.py --model-name deepseek-r1:8b
"""

from langchain_ollama import ChatOllama
from chromadb import Client, Settings
import chromadb
from datetime import datetime
import json

class VectorMemoryManager:
    def __init__(self, model_name="deepseek-r1:8b"):
        print("\n=== INITIALIZING MEMORY SYSTEM ===")
        print("- Creating Ollama chat model")
        self.model = ChatOllama(model=model_name)
        
        print("- Initializing ChromaDB")
        self.db = Client(Settings(allow_reset=True))
        
        print("- Creating memory collections:")
        print("  • User Memories (background, preferences)")
        print("  • Conversation Memories (discussion flow)")
        print("  • Fact Memories (knowledge shared)")
        self.user_memories = self.db.create_collection("user_memories")
        self.conversation_memories = self.db.create_collection("conversation_memories")
        self.fact_memories = self.db.create_collection("fact_memories")
        print("=== MEMORY SYSTEM READY ===\n")
        
    def chat(self, user_input):
        """Process chat with vector-based memory management"""
        print("\n" + "="*70)
        print("CHAT PROCESS STARTED")
        print("="*70)
        
        # STEP 1: Memory Retrieval
        print("\n" + "="*50)
        print("STEP 1: SEARCHING RELEVANT MEMORIES")
        print("="*50)
        print(f"Searching memories related to: '{user_input}'")
        relevant_memories = self._get_relevant_memories(user_input)
        
        print("\nRELEVANT MEMORIES FOUND:")
        print("-"*30)
        for mem_type, memories in relevant_memories.items():
            print(f"\n📎 {mem_type}:")
            if memories:
                for mem in memories:
                    print(f"  • {mem}")
            else:
                print("  • No memories yet")
        
        # STEP 2: Response Generation
        print("\n" + "="*50)
        print("STEP 2: GENERATING AI RESPONSE")
        print("="*50)
        print("Combining memories with current input...")
        response = self._get_response(user_input, relevant_memories)
        print("\nAI Response:")
        print("-"*30)
        print(response)
        
        # STEP 3: Memory Updates
        print("\n" + "="*50)
        print("STEP 3: CREATING NEW MEMORIES")
        print("="*50)
        print("Analyzing exchange to create new memories...")
        self._update_memories(user_input, response)
        
        print("\n" + "="*70)
        print("CHAT PROCESS COMPLETED")
        print("="*70)
        
        return response
        
    def _get_relevant_memories(self, user_input):
        """Retrieve relevant memories using semantic search"""
        print("\nSearching each memory collection...")
        relevant_memories = {
            "User Information": self.user_memories.query(
                query_texts=[user_input],
                n_results=3
            ).get('documents', [[]])[0],
            
            "Conversation History": self.conversation_memories.query(
                query_texts=[user_input],
                n_results=3
            ).get('documents', [[]])[0],
            
            "Facts & Knowledge": self.fact_memories.query(
                query_texts=[user_input],
                n_results=3
            ).get('documents', [[]])[0]
        }
        return relevant_memories
        
    def _get_response(self, user_input, memories):
        """Generate response using retrieved memories"""
        context = self._format_memories_for_context(memories)
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful AI assistant. Use these memories for context:

{context}

Respond naturally to the user's message, using the memories to maintain conversation continuity."""
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
        
        response = self.model.invoke(messages)
        return response.content
        
    def _update_memories(self, user_input, ai_response):
        """Update vector database with new memories"""
        print("\nGenerating new memories from exchange...")
        memory_updates = self._generate_memory_updates(user_input, ai_response)
        
        print("\n=== NEW MEMORIES CREATED ===")
        
        # Update user memories
        if memory_updates.get("user"):
            print("\n📌 NEW USER MEMORY:")
            print("-"*30)
            print(memory_updates["user"])
            self.user_memories.add(
                documents=[memory_updates["user"]],
                ids=[f"user_mem_{datetime.now().timestamp()}"],
                metadatas=[{"type": "user", "timestamp": str(datetime.now())}]
            )
            
        # Update conversation memories
        if memory_updates.get("conversation"):
            print("\n📌 NEW CONVERSATION MEMORY:")
            print("-"*30)
            print(memory_updates["conversation"])
            self.conversation_memories.add(
                documents=[memory_updates["conversation"]],
                ids=[f"conv_mem_{datetime.now().timestamp()}"],
                metadatas=[{"type": "conversation", "timestamp": str(datetime.now())}]
            )
            
        # Update fact memories
        if memory_updates.get("facts"):
            print("\n📌 NEW FACT MEMORY:")
            print("-"*30)
            print(memory_updates["facts"])
            self.fact_memories.add(
                documents=[memory_updates["facts"]],
                ids=[f"fact_mem_{datetime.now().timestamp()}"],
                metadatas=[{"type": "fact", "timestamp": str(datetime.now())}]
            )
        
        print("\n=== MEMORIES STORED IN DATABASE ===")

    def _generate_memory_updates(self, user_input, ai_response):
        """Generate structured memory updates"""
        prompt = f"""Analyze this exchange and create three types of memories:

Exchange:
User: {user_input}
Assistant: {ai_response}

Create three different types of memories:
1. User Memory: Information about the user's background, preferences, and traits
2. Conversation Memory: The flow, context, and development of the discussion
3. Fact Memory: Specific facts, information, or knowledge shared

Format as JSON with these keys: "user", "conversation", "facts"
"""
        
        response = self.model.invoke([{"role": "system", "content": prompt}])
        try:
            memories = json.loads(response.content)
            return memories
        except:
            return {
                "user": "Failed to parse user memory",
                "conversation": "Failed to parse conversation memory",
                "facts": "Failed to parse fact memory"
            }
            
    def _format_memories_for_context(self, memories):
        """Format retrieved memories into context string"""
        context_parts = []
        for memory_type, memory_list in memories.items():
            if memory_list:
                context_parts.append(f"{memory_type}:")
                context_parts.extend([f"- {mem}" for mem in memory_list])
        
        return "\n\n".join(context_parts)
        
    def clear(self):
        """Clear all memories"""
        print("\n=== CLEARING ALL MEMORIES ===")
        self.db.reset()
        print("✓ User memories cleared")
        print("✓ Conversation memories cleared")
        print("✓ Fact memories cleared")
        print("=== MEMORY SYSTEM RESET ===")

class MemoryManagerCLI:
    def __init__(self, model_name="deepseek-r1:8b"):
        self.memory = VectorMemoryManager(model_name=model_name)

    def run(self):
        """Main CLI loop"""
        print("\nVector Memory Manager")
        print("Enter messages or commands (/help for list)")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue
                
                if user_input:
                    self.memory.chat(user_input)
                    
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
        else:
            print(f"Unknown command: {command}")

    def _print_help(self):
        """Show available commands"""
        print("\nCommands:")
        print("  /quit   - Exit the program")
        print("  /clear  - Clear all memories")
        print("  /help   - Show this help message")

def main():
    """Start the CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vector Memory Manager')
    parser.add_argument('--model-name', default='deepseek-r1:8b', help='Name of the Ollama model to use')
    
    args = parser.parse_args()
    
    cli = MemoryManagerCLI(model_name=args.model_name)
    cli.run()

if __name__ == "__main__":
    main() 