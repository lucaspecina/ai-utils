#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tool-Using Model Inference Example

This script demonstrates how to use a fine-tuned tool-using model for inference.
"""

import json
import argparse
import re
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Define tool schemas (same as in training script)
TOOLS = [
    {
        "name": "search_web",
        "description": "Search the web for information on a topic",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or location name"
                },
                "units": {
                    "type": "string",
                    "description": "Temperature units (celsius/fahrenheit)"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    }
]


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from the model's generated text."""
    tool_calls = []
    
    # Regular expression to match tool calls
    pattern = r'<tool_call>\s*name:\s*([^\n]+)\s*parameters:\s*({[^<]+})\s*</tool_call>'
    
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        tool_name = match.group(1).strip()
        parameters_str = match.group(2).strip()
        
        try:
            # Parse the parameters as JSON
            parameters = json.loads(parameters_str)
            
            tool_calls.append({
                "name": tool_name,
                "parameters": parameters
            })
        except json.JSONDecodeError:
            print(f"Warning: Could not parse parameters for tool call: {tool_name}")
            continue
    
    return tool_calls


def simulate_tool_execution(tool_call: Dict[str, Any]) -> str:
    """Simulate executing a tool call (for demonstration purposes)."""
    tool_name = tool_call["name"]
    params = tool_call["parameters"]
    
    if tool_name == "search_web":
        query = params.get("query", "")
        return f"[Simulated search results for '{query}']:\n- Result 1: Example information about {query}\n- Result 2: More details about {query}"
    
    elif tool_name == "get_weather":
        location = params.get("location", "")
        return f"[Simulated weather for {location}]: 22°C, Partly Cloudy, Humidity: 65%, Wind: 10 km/h"
    
    elif tool_name == "calculate":
        expression = params.get("expression", "")
        return f"[Simulated calculation result for '{expression}']: 42"
    
    return f"[Unknown tool: {tool_name}]"


def format_prompt(user_query: str, system_prompt: Optional[str] = None) -> str:
    """Format the prompt for the model."""
    default_system_prompt = (
        "You are a helpful AI assistant that can use tools to answer user questions. "
        "When a tool is needed, you should call it with the appropriate parameters. "
        "Always be helpful, accurate, and concise."
    )
    
    system_prompt = system_prompt or default_system_prompt
    
    # Add tool descriptions
    system_prompt += "\n\nYou have access to the following tools:\n"
    for tool in TOOLS:
        system_prompt += f"- {tool['name']}: {tool['description']}\n"
    
    # Format the full prompt
    formatted_prompt = f"<|system|>\n{system_prompt}\n\n<|user|>\n{user_query}\n\n<|assistant|>"
    
    return formatted_prompt


def generate_response(
    model_path: str,
    user_query: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    system_prompt: Optional[str] = None,
    base_model_name: Optional[str] = None
) -> str:
    """Generate a response from the fine-tuned model."""
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Check if this is a PEFT model
    try:
        config = PeftConfig.from_pretrained(model_path)
        # Use provided base_model_name if available, otherwise use the one from config
        base_model_path = base_model_name or config.base_model_name_or_path
        print(f"Loading PEFT model with base model: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    except:
        print(f"Loading full model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Format the prompt
    prompt = format_prompt(user_query, system_prompt)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return response


def main():
    """Main function to run the inference example."""
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned tool-using model")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model")
    parser.add_argument("--query", type=str, default="What's the weather like in Paris?",
                        help="User query to process")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--execute_tools", action="store_true",
                        help="Whether to execute (simulate) the tool calls")
    
    args = parser.parse_args()
    
    print(f"User query: {args.query}")
    print("-" * 50)
    
    # Generate initial response
    response = generate_response(
        model_path=args.model_path,
        user_query=args.query,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    print(f"Model response:\n{response}")
    print("-" * 50)
    
    # Extract tool calls
    tool_calls = extract_tool_calls(response)
    
    if tool_calls:
        print(f"Extracted {len(tool_calls)} tool call(s):")
        for i, tool_call in enumerate(tool_calls):
            print(f"Tool call {i+1}:")
            print(f"  Name: {tool_call['name']}")
            print(f"  Parameters: {json.dumps(tool_call['parameters'], indent=2)}")
            
            if args.execute_tools:
                # Simulate tool execution
                tool_response = simulate_tool_execution(tool_call)
                print(f"Tool response:\n{tool_response}")
                
                # Generate follow-up response with tool results
                follow_up_query = f"{args.query}\n\nTool response: {tool_response}"
                follow_up_response = generate_response(
                    model_path=args.model_path,
                    user_query=follow_up_query,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature
                )
                
                print("-" * 50)
                print(f"Follow-up response:\n{follow_up_response}")
    else:
        print("No tool calls detected in the response.")


if __name__ == "__main__":
    main() 