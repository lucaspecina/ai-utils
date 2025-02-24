"""
A CLI chat interface using HuggingFace Transformers for local inference.
Usage: python inference/conversation-transformers.py [--model MODEL] [--device DEVICE]

Examples:
    # Start chat with default model on CPU
    python inference/conversation-transformers.py

    # Use specific model on GPU
    python inference/conversation-transformers.py --model facebook/opt-350m --device cuda

    # Use quantized model for better memory efficiency
    python inference/conversation-transformers.py --model TheBloke/Mistral-7B-v0.1-GGUF --device cpu --quantized
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer
)
from threading import Thread
from typing import List, Optional
import torch
import argparse
import sys
from queue import Queue
import threading

class LocalChatAI:
    """Handles chat interactions using local transformer models"""
    
    def __init__(
        self,
        model_name: str = "facebook/opt-350m",
        device: str = "cpu",
        max_length: int = 2048,
        quantized: bool = False
    ):
        """Initialize the chat model
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cpu' or 'cuda'
            max_length: Maximum token length for generation
            quantized: Whether to use quantization for reduced memory
        """
        print(f"Loading model {model_name}...")
        
        # Set up device
        self.device = device
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with optional quantization
        if quantized:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                quantization_config=quantization_config
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            if device == "cuda":
                self.model = self.model.to(device)
        
        # Set up generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )
        
        # Initialize chat history and streamer
        self.chat_history: List[str] = []
        self.streamer = TextIteratorStreamer(self.tokenizer)
        self.output_queue = Queue()

    def _build_prompt(self, user_input: str) -> str:
        """Build the complete prompt including chat history"""
        prompt = "You are a helpful AI assistant. Respond thoughtfully to the user's questions.\n\n"
        
        # Add chat history
        for message in self.chat_history:
            prompt += message + "\n"
            
        # Add current input
        prompt += f"User: {user_input}\nAssistant:"
        return prompt

    def _stream_output(self, prompt: str):
        """Generate response in a separate thread"""
        self.generator(
            prompt,
            max_length=self.max_length,
            num_return_sequences=1,
            streamer=self.streamer,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )

    def chat(self, user_input: str) -> str:
        """Process a chat interaction with streaming output
        
        Args:
            user_input: The user's message
        Returns:
            The AI's response
        """
        # Build the complete prompt
        prompt = self._build_prompt(user_input)
        
        # Start generation in a separate thread
        thread = Thread(target=self._stream_output, args=(prompt,))
        thread.start()
        
        # Collect and display streaming output
        collected_output = []
        print("\nAI:", end=" ", flush=True)
        
        for text in self.streamer:
            print(text, end="", flush=True)
            collected_output.append(text)
        
        print()  # New line after response
        
        # Join the output and clean it
        response = "".join(collected_output)
        response = response.replace(prompt, "").strip()
        
        # Update chat history
        self.chat_history.append(f"User: {user_input}")
        self.chat_history.append(f"Assistant: {response}")
        
        return response

    def clear_history(self):
        """Reset the conversation history"""
        self.chat_history = []

    def display_history(self):
        """Show the conversation history"""
        if not self.chat_history:
            print("No chat history available.")
            return
        
        print("\nChat History:")
        for message in self.chat_history:
            print(message)

def print_help():
    """Display available commands"""
    print("\nCommands:")
    print("  /quit    - Exit the chat")
    print("  /clear   - Clear chat history")
    print("  /history - Show chat history")
    print("  /help    - Show this help message")

def main():
    """Main chat loop"""
    parser = argparse.ArgumentParser(description='Chat with local transformer model')
    parser.add_argument(
        '--model',
        default='facebook/opt-350m',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run the model on'
    )
    parser.add_argument(
        '--quantized',
        action='store_true',
        help='Use 4-bit quantization for reduced memory usage'
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability if requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    try:
        chat_ai = LocalChatAI(
            model_name=args.model,
            device=args.device,
            quantized=args.quantized
        )
        
        print(f"\nChat started with {args.model} on {args.device}")
        print("Type messages or commands (/help for list)")
        
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
                    elif command == 'help':
                        print_help()
                    else:
                        print(f"Unknown command: {user_input}")
                    continue
                
                if user_input:
                    chat_ai.chat(user_input)
                    
            except KeyboardInterrupt:
                print("\nUse /quit to exit")
                continue
                
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 