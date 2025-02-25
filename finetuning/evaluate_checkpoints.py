#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Checkpoint Evaluation Script

This script evaluates a model at different checkpoints during training.
It compares the base model with fine-tuned checkpoints on tool-using tasks.
"""

import os
import json
import argparse
import logging
import glob
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from datasets import Dataset, load_dataset

# Import from inference example
from inference_example import generate_response, extract_tool_calls, simulate_tool_execution

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


def find_checkpoints(checkpoint_dir: str) -> List[str]:
    """Find all checkpoint directories in the given directory."""
    # Look for checkpoint directories
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    
    return checkpoints


def evaluate_checkpoint(
    checkpoint_path: str,
    test_examples: List[Dict[str, Any]],
    base_model_name: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """Evaluate a checkpoint on test examples."""
    results = {
        "checkpoint": checkpoint_path,
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
    
    for example in test_examples:
        user_query = example["query"]
        expected_tool = example["expected_tool"]
        
        # Generate response
        response = generate_response(
            model_path=checkpoint_path,
            user_query=user_query,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            base_model_name=base_model_name
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


def generate_response_with_base_model(
    checkpoint_path: str,
    base_model_name: str,
    user_query: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1
) -> str:
    """Generate a response using a checkpoint with its base model."""
    # Load PEFT config to get base model
    config = PeftConfig.from_pretrained(checkpoint_path)
    
    # If base_model_name is provided, use it instead
    base_model_name = base_model_name or config.base_model_name_or_path
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Format prompt
    from inference_example import format_prompt
    prompt = format_prompt(user_query)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return response


def main():
    """Main function to run the checkpoint evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints on tool-using tasks")
    
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing model checkpoints")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset file")
    parser.add_argument("--base_model_name", type=str, default=None,
                        help="Name of the base model (if not specified in PEFT config)")
    parser.add_argument("--num_examples", type=int, default=10,
                        help="Number of examples to evaluate")
    parser.add_argument("--output_file", type=str, default="checkpoint_evaluation.json",
                        help="Path to save evaluation results")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Load test examples
    logger.info(f"Loading test examples from {args.data_path}")
    test_examples = load_test_examples(args.data_path, args.num_examples)
    logger.info(f"Loaded {len(test_examples)} test examples")
    
    # Find checkpoints
    logger.info(f"Finding checkpoints in {args.checkpoint_dir}")
    checkpoints = find_checkpoints(args.checkpoint_dir)
    logger.info(f"Found {len(checkpoints)} checkpoints")
    
    # Add the final model if it exists
    final_model_path = args.checkpoint_dir
    if os.path.exists(os.path.join(final_model_path, "adapter_config.json")):
        checkpoints.append(final_model_path)
        logger.info(f"Added final model: {final_model_path}")
    
    # Evaluate each checkpoint
    all_results = []
    
    for checkpoint in checkpoints:
        logger.info(f"Evaluating checkpoint: {checkpoint}")
        
        try:
            results = evaluate_checkpoint(
                checkpoint_path=checkpoint,
                test_examples=test_examples,
                base_model_name=args.base_model_name,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
            
            # Log metrics
            metrics = results["metrics"]
            logger.info(f"Tool call accuracy: {metrics['tool_call_accuracy']:.2f}")
            logger.info(f"Parameter accuracy: {metrics['parameter_accuracy']:.2f}")
            logger.info(f"Total tool calls: {metrics['total_tool_calls']}")
            
            all_results.append(results)
        except Exception as e:
            logger.error(f"Error evaluating checkpoint {checkpoint}: {e}")
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {args.output_file}")


if __name__ == "__main__":
    main() 