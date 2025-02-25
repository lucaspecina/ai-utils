#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ollama Tool-Using Evaluation Script

This script evaluates Llama 3.1 in Ollama on tool-using tasks.
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Import Ollama adapter
from ollama_adapter import (
    generate_response_with_ollama,
    extract_tool_calls,
    simulate_tool_execution,
    test_ollama_setup
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_test_examples(data_path: str, num_examples: int = 10) -> List[Dict[str, Any]]:
    """Load test examples from a dataset file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Take a sample of examples
    examples = data[:num_examples]
    
    # Extract user queries
    test_examples = []
    for example in examples:
        user_query = example["messages"][0]["content"]
        expected_tool = None
        
        # Find the expected tool call
        for message in example["messages"]:
            if message["role"] == "assistant" and "tool_calls" in message:
                expected_tool = message["tool_calls"][0]
                break
        
        test_examples.append({
            "query": user_query,
            "expected_tool": expected_tool
        })
    
    return test_examples


def evaluate_ollama_model(
    test_examples: List[Dict[str, Any]],
    model_name: str = "llama3.1:latest",
    temperature: float = 0.1,
    max_tokens: int = 1024
) -> Dict[str, Any]:
    """Evaluate the Ollama model on test examples."""
    results = {
        "model": model_name,
        "examples": [],
        "metrics": {
            "tool_call_accuracy": 0.0,
            "parameter_accuracy": 0.0,
            "total_tool_calls": 0
        }
    }
    
    correct_tool_calls = 0
    correct_parameters = 0
    total_tool_calls = 0
    
    for example in tqdm(test_examples, desc=f"Evaluating {model_name}"):
        user_query = example["query"]
        expected_tool = example["expected_tool"]
        
        # Generate response
        response = generate_response_with_ollama(
            user_query=user_query,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract tool calls
        tool_calls = extract_tool_calls(response)
        
        example_result = {
            "query": user_query,
            "response": response,
            "expected_tool": expected_tool,
            "actual_tools": tool_calls,
            "tool_call_correct": False,
            "parameters_correct": False
        }
        
        # Check if tool call is correct
        if tool_calls and expected_tool:
            total_tool_calls += 1
            actual_tool = tool_calls[0]
            
            # Check if tool name is correct
            if actual_tool["name"] == expected_tool["name"]:
                correct_tool_calls += 1
                example_result["tool_call_correct"] = True
                
                # Check if parameters are correct
                expected_params = set(expected_tool["parameters"].keys())
                actual_params = set(actual_tool["parameters"].keys())
                
                # Check required parameters
                required_params_correct = all(
                    param in actual_params and actual_tool["parameters"][param] == expected_tool["parameters"][param]
                    for param in expected_params if param in expected_tool["parameters"]
                )
                
                if required_params_correct:
                    correct_parameters += 1
                    example_result["parameters_correct"] = True
        
        results["examples"].append(example_result)
    
    # Calculate metrics
    if total_tool_calls > 0:
        results["metrics"]["tool_call_accuracy"] = correct_tool_calls / total_tool_calls
        results["metrics"]["parameter_accuracy"] = correct_parameters / total_tool_calls
    results["metrics"]["total_tool_calls"] = total_tool_calls
    
    return results


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Llama 3.1 in Ollama on tool-using tasks")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="llama3.1:latest",
                        help="Model to evaluate in Ollama")
    parser.add_argument("--num_examples", type=int, default=10,
                        help="Number of examples to evaluate")
    parser.add_argument("--output_file", type=str, default="ollama_evaluation.json",
                        help="Path to save evaluation results")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Maximum number of tokens to generate")
    
    args = parser.parse_args()
    
    # Test Ollama setup
    if not test_ollama_setup():
        logger.error("Ollama setup check failed. Please make sure Ollama is running and the model is available.")
        return
    
    # Load test examples
    logger.info(f"Loading test examples from {args.data_path}")
    test_examples = load_test_examples(args.data_path, args.num_examples)
    logger.info(f"Loaded {len(test_examples)} test examples")
    
    # Evaluate model
    logger.info(f"Evaluating model: {args.model}")
    results = evaluate_ollama_model(
        test_examples=test_examples,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Log metrics
    metrics = results["metrics"]
    logger.info(f"Tool call accuracy: {metrics['tool_call_accuracy']:.2f}")
    logger.info(f"Parameter accuracy: {metrics['parameter_accuracy']:.2f}")
    logger.info(f"Total tool calls: {metrics['total_tool_calls']}")
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {args.output_file}")


if __name__ == "__main__":
    main() 