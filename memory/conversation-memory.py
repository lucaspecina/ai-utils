"""
Memory management system for AI conversations with automatic summarization and CLI interface.

Basic Usage:
    # Use with Ollama (default)
    python conversation_memory.py
    
    # Use with specific Ollama model
    python conversation_memory.py --model-name codellama
    
    # Use with local transformers model
    python conversation_memory.py --model-type transformers --model-name facebook/opt-1.3b

Commands:
    /summary    - Show current conversation memory
    /context    - Show full context being used
    /clear      - Clear conversation history
    /save file  - Save conversation to file
    /load file  - Load conversation from file
    /quit       - Exit the program
"""

from langchain_ollama import ChatOllama
import json

class ConversationMemory:
    def __init__(self, model_name="deepseek-r1:8b"):
        self.model = ChatOllama(model=model_name)
        self.memory = ""
        
    def chat(self, user_input):
        """Process chat with clear steps:
        1. Show current memory
        2. Get AI response using memory
        3. Update memory with new information
        """
        # STEP 1: Show current memory
        print("\n" + "="*50)
        print("STEP 1: CURRENT MEMORY")
        print("="*50)
        if self.memory:
            print(self.memory)
        else:
            print("No memories yet")
        
        # STEP 2: Get AI response
        print("\n" + "="*50)
        print("STEP 2: GETTING AI RESPONSE")
        print("="*50)
        response = self._get_response(user_input)
        print(f"Assistant: {response}")
        
        # STEP 3: Update memory
        print("\n" + "="*50)
        print("STEP 3: UPDATING MEMORY")
        print("="*50)
        self._update_memory(user_input, response)
        
        return response
    
    def _get_response(self, user_input):
        """Get AI response using memory + current question as context"""
        messages = [
            {
                "role": "system",
                "content": """You are a helpful AI assistant. Use the provided memory to maintain conversation continuity.
                
MEMORY OF OUR CONVERSATION:
{self.memory if self.memory else "This is our first interaction."}

Remember to use this memory to inform your response, but respond naturally to the current question."""
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
        
        response = self.model.invoke(messages)
        return response.content
    
    def _update_memory(self, user_input, ai_response):
        """Create comprehensive memories about the entire conversation history"""
        messages = [
            {
                "role": "system",
                "content": f"""You are the memory system of an AI assistant. Your role is to observe and remember 
everything important about the conversations. Think of yourself as the AI's long-term memory.

CURRENT MEMORY:
{self.memory if self.memory else "This is our first interaction with this user."}

NEW INTERACTION:
User: {user_input}
Assistant: {ai_response}

Create a rich, detailed memory that captures:

1. About the User:
   - Their background (e.g., being from Argentina)
   - Their interests and preferences (e.g., interest in tango)
   - How they express themselves
   - What seems to matter to them

2. About the Assistant's Responses:
   - How it's been helping the user
   - Key information it has shared
   - The approach it's taking
   - The relationship being built

3. About the Conversation:
   - Main topics discussed
   - Important facts revealed
   - The flow of discussion
   - Notable moments or insights

Write this memory as a clear narrative that will help the assistant maintain continuity 
and understanding throughout the conversation. Remember EVERYTHING important about both 
the user and how the assistant has been interacting with them."""
            }
        ]
        
        new_memory = self.model.invoke(messages).content
        self.memory = new_memory
        
        print("\n=== MEMORY SYSTEM UPDATE ===")
        print("="*50)
        print(new_memory)
        print("="*50)

    def clear(self):
        """Clear the memory"""
        self.memory = ""
        print("Memory cleared")

class MemoryManagerCLI:
    def __init__(self, model_name="deepseek-r1:8b"):
        self.memory = ConversationMemory(model_name=model_name)

    def run(self):
        """Main CLI loop"""
        print("\nConversation Memory Manager")
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
        print("  /clear  - Clear conversation memory")
        print("  /help   - Show this help message")

def main():
    """Start the CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Conversation Memory Manager')
    parser.add_argument('--model-name', default='deepseek-r1:8b', help='Name of the Ollama model to use')
    
    args = parser.parse_args()
    
    cli = MemoryManagerCLI(model_name=args.model_name)
    cli.run()

if __name__ == "__main__":
    main() 