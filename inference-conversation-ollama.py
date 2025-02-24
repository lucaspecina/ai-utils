""" 
python .\inference-conversation-ollama.py --model deepseek-r1:8b
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List
import argparse

class ConversationalAI:
    def __init__(
        self, 
        model_name: str = "deepseek-r1:8b", 
        temperature: float = 0.2,
        max_tokens: int = 1000,
        system_prompt: str = """You are an AI assistant engaged in a conversation. 
You will always respond as the AI, never as the user.
The user's messages will be prefixed with 'You: ' and your responses will be prefixed with 'AI: '.
Maintain a helpful, professional tone and provide accurate, thoughtful responses."""
    ):
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.chat_history: List[HumanMessage | AIMessage] = []
        # Always start with a system message
        self.system_message = SystemMessage(content=system_prompt)

    def chat(self, user_input: str) -> str:
        # Combine system message with chat history and new input
        messages = [self.system_message] + self.chat_history + [HumanMessage(content=user_input)]
        
        # Get response from LLM
        response = self.llm.invoke(messages)
        
        # Only add user message and AI response to history (not system message)
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response.content))
        
        return response.content

    def clear_history(self):
        self.chat_history = []

    def display_history(self):
        for message in self.chat_history:
            role = "User" if isinstance(message, HumanMessage) else "AI"
            print(f"{role}: {message.content}")




def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Chat with an Ollama LLM model')
    parser.add_argument(
        '--model', 
        type=str, 
        default='deepseek-r1:8b',
        help='Name of the Ollama model to use (default: deepseek-r1:8b)'
    )
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.2,
        help='Temperature for response generation (0.0-1.0, default: 0.2)'
    )
    parser.add_argument(
        '--max-tokens', 
        type=int, 
        default=1000,
        help='Maximum tokens in response (default: 1000)'
    )
    parser.add_argument(
        '--system-prompt', 
        type=str,
        default="""You are an AI assistant engaged in a conversation. 
You will always respond as the AI, never as the user.
The user's messages will be prefixed with 'You: ' and your responses will be prefixed with 'AI: '.
Maintain a helpful, professional tone and provide accurate, thoughtful responses.""",
        help='System prompt to set the AI\'s behavior'
    )

    args = parser.parse_args()

    # Initialize the conversational AI with command line arguments
    chat_ai = ConversationalAI(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt
    )
    
    print(f"\nChat started with model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    if args.system_prompt:
        print(f"System prompt: {args.system_prompt}")
    print("\nCommands:")
    print("  'quit' to exit")
    print("  'clear' to clear history")
    print("  'history' to show chat history")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            chat_ai.clear_history()
            print("Chat history cleared!")
            continue
        elif user_input.lower() == 'history':
            chat_ai.display_history()
            continue
            
        response = chat_ai.chat(user_input)
        print(f"\nAI: {response}")

if __name__ == "__main__":
    main()