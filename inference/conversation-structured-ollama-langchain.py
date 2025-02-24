"""
A CLI chat interface using Langchain's structured output parsers.
Usage: python inference/conversation-structured-ollama-langchain.py [--model MODEL] [--temperature TEMP]

Examples:
    # Start chat with default model
    python inference/conversation-structured-ollama-langchain.py

    # Use specific model and temperature
    python inference/conversation-structured-ollama-langchain.py --model codellama --temperature 0.2

Once started, use commands like:
    /json     - Switch to JSON structured output
    /person   - Switch to person info structure
    /list     - Switch to list output
    /free     - Switch to free-form text
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.output_parsers import (
    PydanticOutputParser,
    CommaSeparatedListOutputParser,
    StructuredOutputParser,
    ResponseSchema
)
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import argparse
import json

class PersonInfo(BaseModel):
    """Pydantic model for structured person information"""
    name: str = Field(description="The person's full name")
    occupation: str = Field(description="Primary occupation or role")
    achievements: List[str] = Field(description="List of major achievements")
    impact: str = Field(description="Description of their impact or significance")
    dates: Dict[str, str] = Field(description="Key dates (birth, death, major events)")

class StructuredChatAI:
    """Handles chat interactions with structured output formatting"""
    
    def __init__(
        self, 
        model_name: str = "deepseek-r1:8b", 
        temperature: float = 0.1
    ):
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )
        self.chat_history: List[HumanMessage | AIMessage] = []
        self.current_parser: Optional[Any] = None
        self.current_format: Optional[str] = None
        
        # Initialize parsers
        self.parsers = {
            'json': self._create_json_parser(),
            'person': PydanticOutputParser(pydantic_object=PersonInfo),
            'list': CommaSeparatedListOutputParser()
        }

    def _create_json_parser(self):
        """Create JSON structure parser"""
        response_schemas = [
            ResponseSchema(name="summary", description="Brief summary of the topic"),
            ResponseSchema(name="details", description="Detailed information about the topic"),
            ResponseSchema(name="key_points", description="List of key points or facts"),
            ResponseSchema(name="references", description="Related topics or references")
        ]
        return StructuredOutputParser.from_response_schemas(response_schemas)

    def get_format_instructions(self, format_type: str) -> str:
        """Get system instructions for different format types"""
        if format_type == 'json':
            return """Provide structured information with:
- summary: Brief overview
- details: Comprehensive explanation
- key_points: Important facts
- references: Related topics"""
        
        elif format_type == 'person':
            return """Provide person information with:
- name: Full name
- occupation: Primary role
- achievements: List of accomplishments
- impact: Historical significance
- dates: Key dates dictionary"""
        
        elif format_type == 'list':
            return "Provide a comma-separated list of items"
        
        return "Provide a natural, conversational response"

    def set_format(self, format_type: str) -> str:
        """Set the current output format"""
        if format_type in self.parsers or format_type == 'free':
            self.current_format = format_type
            self.current_parser = self.parsers.get(format_type)
            return f"Switched to {format_type} format"
        return f"Unknown format: {format_type}"

    def chat(self, user_input: str) -> Any:
        """Process a chat interaction with optional format parsing"""
        
        # Create system message with current format instructions
        system_msg = SystemMessage(content=f"""
You are an AI that provides structured responses.
{self.get_format_instructions(self.current_format or 'free')}
{self.current_parser.get_format_instructions() if self.current_parser else ''}
        """)
        
        # Combine messages
        messages = [system_msg] + self.chat_history + [HumanMessage(content=user_input)]
        
        # Get response
        response = self.llm.invoke(messages)
        
        # Store the interaction
        self.chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response.content)
        ])
        
        # Parse response if parser is set
        if self.current_parser:
            try:
                return self.current_parser.parse(response.content)
            except Exception as e:
                return {"error": f"Failed to parse response: {str(e)}"}
        
        return response.content

    def clear_history(self):
        """Reset the conversation history"""
        self.chat_history = []

    def display_history(self):
        """Show the conversation history"""
        if not self.chat_history:
            print("No chat history available.")
            return
            
        for message in self.chat_history:
            prefix = "You:" if isinstance(message, HumanMessage) else "AI:"
            print(f"\n{prefix} {message.content}")

def print_help():
    """Display available commands"""
    print("\nCommands:")
    print("  /quit    - Exit the chat")
    print("  /clear   - Clear chat history")
    print("  /history - Show chat history")
    print("  /json    - Switch to JSON format")
    print("  /person  - Switch to person info format")
    print("  /list    - Switch to list format")
    print("  /free    - Switch to free-form text")
    print("  /help    - Show this help message")

def main():
    """Main chat loop"""
    parser = argparse.ArgumentParser(description='Chat with structured outputs')
    parser.add_argument('--model', default='deepseek-r1:8b', help='Ollama model name')
    parser.add_argument('--temperature', type=float, default=0.1, help='Response randomness')
    args = parser.parse_args()

    chat_ai = StructuredChatAI(
        model_name=args.model,
        temperature=args.temperature
    )
    
    print(f"\nStructured Chat started with {args.model}")
    print("Type messages normally or use commands for specific formats.")
    print_help()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.startswith('/'):
                command = user_input[1:].lower()
                if command == 'quit':
                    print("Goodbye!")
                    break
                elif command == 'clear':
                    chat_ai.clear_history()
                    print("Chat history cleared!")
                elif command == 'history':
                    chat_ai.display_history()
                elif command in ['json', 'person', 'list', 'free']:
                    print(chat_ai.set_format(command))
                elif command == 'help':
                    print_help()
                else:
                    print(f"Unknown command: {user_input}")
                continue
            
            if user_input:
                response = chat_ai.chat(user_input)
                if isinstance(response, (dict, list)):
                    print("\nAI:", json.dumps(response, indent=2))
                else:
                    print("\nAI:", response)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 