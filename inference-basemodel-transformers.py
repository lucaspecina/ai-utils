"""
A CLI tool for text completion using base LLMs (not instruction-tuned).
Usage: python inference-basemodel-transformers.py [--model MODEL] [--device DEVICE]

Examples:
    # Use base GPT-2
    python inference-basemodel-transformers.py --model gpt2-medium

    # Use base LLAMA
    python inference-basemodel-transformers.py --model facebook/opt-1.3b --device cuda

    # Use with specific settings
    python inference-basemodel-transformers.py --model EleutherAI/pythia-1.4b --max_length 200 --temp 0.9
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Optional
import argparse
from threading import Thread
import sys

class BaseModelLLM:
    """Handles text completion with base language models (pre-instruction-tuning)"""
    
    def __init__(
        self,
        model_name: str = "gpt2-medium",
        device: str = "cpu",
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        use_half_precision: bool = True
    ):
        """Initialize the base language model"""
        print(f"Loading base model {model_name}...")
        
        # Set up device
        if device.startswith('cuda') and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
        self.device = torch.device(device)
        
        # Configure model precision
        self.dtype = torch.float16 if use_half_precision and device.startswith('cuda') else torch.float32
        
        # Load tokenizer with padding token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        try:
            # Try loading with GPU optimizations
            model_kwargs = {
                "torch_dtype": self.dtype,
            }
            
            if device.startswith('cuda'):
                try:
                    import accelerate
                    # Add GPU optimizations if Accelerate is available
                    model_kwargs.update({
                        "device_map": device,
                        "low_cpu_mem_usage": True,
                    })
                except ImportError:
                    print("Accelerate library not found. Installing basic GPU support.")
                    # Fallback to basic GPU support
                    model_kwargs.update({
                        "device_map": None,
                    })
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Move model to device if not using device_map
            if not device.startswith('cuda') or 'device_map' not in model_kwargs:
                self.model = self.model.to(self.device)
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Try installing Accelerate: pip install 'accelerate>=0.26.0'")
            raise
        
        # Update generation config
        self.generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,  # Enable KV-cache for faster generation
        }
        
        # Initialize context
        self.current_context = ""
        
        # Print memory usage if on GPU
        if device.startswith('cuda'):
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated(self.device) / 1024**2:.2f}MB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved(self.device) / 1024**2:.2f}MB")

    def get_raw_completion(
        self,
        prompt: str,
        show_tokens: bool = False,
        continue_previous: bool = False
    ) -> Dict[str, str]:
        """Get raw completion from the model
        
        Args:
            prompt: Input text to complete
            show_tokens: Whether to show token information
            continue_previous: Whether to continue from previous context
        """
        # Build full context
        if continue_previous and self.current_context:
            full_prompt = self.current_context + "\n" + prompt
        else:
            full_prompt = prompt
            
        # Tokenize input
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        input_length = len(inputs.input_ids[0])
        
        if show_tokens:
            print(f"\nInput tokens: {input_length}")
            print("Token sequence:")
            for i, token_id in enumerate(inputs.input_ids[0]):
                print(f"{i}: {token_id} -> {self.tokenizer.decode([token_id])}")
        
        # Generate completion
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_config
            )
        
        # Process all sequences
        completions = []
        for output in outputs:
            # Remove input tokens to get only the completion
            completion_tokens = output[input_length:]
            completion_text = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
            completions.append(completion_text)
            
            # Update context with the first completion
            if len(completions) == 1:
                self.current_context = full_prompt + completion_text
        
        result = {
            "prompt": full_prompt,
            "completions": completions,
        }
        
        if show_tokens:
            result["completion_tokens"] = len(completion_tokens)
            result["total_tokens"] = len(output)
            
        return result

    def clear_context(self):
        """Reset the current context"""
        self.current_context = ""
        print("Context cleared!")

def print_help():
    """Display available commands"""
    print("\nCommands:")
    print("  /quit     - Exit")
    print("  /tokens   - Toggle token information")
    print("  /temp N   - Set temperature (0.0-1.0)")
    print("  /length N - Set max completion length")
    print("  /clear    - Clear current context")
    print("  /context  - Show current context")
    print("  /new      - Start new completion (don't use context)")
    print("  /help     - Show this help message")

def main():
    """Main interaction loop"""
    parser = argparse.ArgumentParser(description='Base Model LLM text completion')
    parser.add_argument(
        '--model',
        default='gpt2-medium',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run the model on (e.g., cpu, cuda, cuda:0)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Maximum completion length'
    )
    parser.add_argument(
        '--temp',
        type=float,
        default=0.8,
        help='Temperature for generation'
    )
    parser.add_argument(
        '--fp32',
        action='store_true',
        help='Use FP32 instead of FP16 on GPU'
    )
    
    args = parser.parse_args()
    
    try:
        llm = BaseModelLLM(
            model_name=args.model,
            device=args.device,
            max_new_tokens=args.max_length,
            temperature=args.temp,
            use_half_precision=not args.fp32
        )
        
        print(f"\nBase Model LLM initialized with {args.model} on {args.device}")
        if args.device.startswith('cuda'):
            print(f"Using {'FP32' if args.fp32 else 'FP16'} precision")
        print("Enter text to see completions")
        print("Text will build upon previous context")
        print("Use /help for commands")
        
        show_tokens = False
        continue_context = True
        
        while True: 
            try:
                user_input = input("\nPrompt: ").strip()
                
                if user_input.startswith('/'):
                    command = user_input[1:].lower().split()
                    cmd = command[0]
                    
                    if cmd == 'quit':
                        break
                    elif cmd == 'tokens':
                        show_tokens = not show_tokens
                        print(f"Token information: {show_tokens}")
                    elif cmd == 'temp' and len(command) > 1:
                        temp = float(command[1])
                        llm.generation_config["temperature"] = temp
                        print(f"Temperature set to {temp}")
                    elif cmd == 'length' and len(command) > 1:
                        length = int(command[1])
                        llm.generation_config["max_new_tokens"] = length
                        print(f"Max length set to {length}")
                    elif cmd == 'clear':
                        llm.clear_context()
                    elif cmd == 'context':
                        print("\nCurrent context:")
                        print(llm.current_context or "(empty)")
                    elif cmd == 'new':
                        continue_context = False
                        print("Starting new completion (ignoring context)")
                    elif cmd == 'help':
                        print_help()
                    else:
                        print(f"Unknown command: {cmd}")
                    continue
                
                if user_input:
                    result = llm.get_raw_completion(
                        user_input,
                        show_tokens=show_tokens,
                        continue_previous=continue_context
                    )
                    
                    print("\nCompletions:")
                    for i, completion in enumerate(result["completions"], 1):
                        print(f"\n--- Completion {i} ---")
                        print(completion)
                    
                    if show_tokens and "completion_tokens" in result:
                        print(f"\nCompletion tokens: {result['completion_tokens']}")
                        print(f"Total tokens: {result['total_tokens']}")
                    
                    # Reset to default behavior after /new
                    continue_context = True
                    
            except KeyboardInterrupt:
                print("\nUse /quit to exit")
                continue
                
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 