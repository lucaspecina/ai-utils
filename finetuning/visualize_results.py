#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization Script for Checkpoint Evaluation Results

This script creates visualizations from the checkpoint evaluation results.
"""

import os
import json
import argparse
import re
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def extract_step_number(checkpoint_path: str) -> int:
    """Extract the step number from a checkpoint path."""
    match = re.search(r'checkpoint-(\d+)', checkpoint_path)
    if match:
        return int(match.group(1))
    return 0  # Final model or unknown


def load_results(results_file: str) -> List[Dict[str, Any]]:
    """Load evaluation results from a JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Sort results by step number
    results.sort(key=lambda x: extract_step_number(x["checkpoint"]))
    
    return results


def plot_accuracy_over_steps(results: List[Dict[str, Any]], output_dir: str) -> None:
    """Plot tool call and parameter accuracy over training steps."""
    steps = [extract_step_number(r["checkpoint"]) for r in results]
    tool_accuracies = [r["metrics"]["tool_call_accuracy"] for r in results]
    param_accuracies = [r["metrics"]["parameter_accuracy"] for r in results]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, tool_accuracies, 'b-o', label='Tool Call Accuracy')
    plt.plot(steps, param_accuracies, 'r-o', label='Parameter Accuracy')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.title('Tool-Using Accuracy Over Training Steps')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add final point label
    if steps and tool_accuracies:
        plt.annotate(f'{tool_accuracies[-1]:.2f}', 
                    xy=(steps[-1], tool_accuracies[-1]),
                    xytext=(5, 5), textcoords='offset points')
        plt.annotate(f'{param_accuracies[-1]:.2f}', 
                    xy=(steps[-1], param_accuracies[-1]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_over_steps.png'), dpi=300)
    plt.close()


def plot_tool_usage(results: List[Dict[str, Any]], output_dir: str) -> None:
    """Plot tool usage distribution for the final checkpoint."""
    if not results:
        return
    
    # Use the last result (final model)
    final_result = results[-1]
    
    # Count tool usage
    tool_counts = {}
    
    for example in final_result["examples"]:
        if "actual_tools" in example and example["actual_tools"]:
            tool_name = example["actual_tools"][0]["name"]
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
    
    if not tool_counts:
        return
    
    # Create bar chart
    tools = list(tool_counts.keys())
    counts = list(tool_counts.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(tools, counts, color='skyblue')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.0f}', ha='center', va='bottom')
    
    plt.xlabel('Tool Name')
    plt.ylabel('Usage Count')
    plt.title('Tool Usage Distribution (Final Model)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'tool_usage_distribution.png'), dpi=300)
    plt.close()


def plot_parameter_accuracy_by_tool(results: List[Dict[str, Any]], output_dir: str) -> None:
    """Plot parameter accuracy by tool for the final checkpoint."""
    if not results:
        return
    
    # Use the last result (final model)
    final_result = results[-1]
    
    # Count correct parameters by tool
    tool_param_accuracy = {}
    
    for example in final_result["examples"]:
        if "actual_tools" in example and example["actual_tools"]:
            tool_name = example["actual_tools"][0]["name"]
            
            if tool_name not in tool_param_accuracy:
                tool_param_accuracy[tool_name] = {"correct": 0, "total": 0}
            
            tool_param_accuracy[tool_name]["total"] += 1
            
            if example["parameters_correct"]:
                tool_param_accuracy[tool_name]["correct"] += 1
    
    if not tool_param_accuracy:
        return
    
    # Calculate accuracy percentages
    tools = []
    accuracies = []
    
    for tool, counts in tool_param_accuracy.items():
        if counts["total"] > 0:
            tools.append(tool)
            accuracies.append(counts["correct"] / counts["total"] * 100)
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(tools, accuracies, color='lightgreen')
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.xlabel('Tool Name')
    plt.ylabel('Parameter Accuracy (%)')
    plt.title('Parameter Accuracy by Tool (Final Model)')
    plt.xticks(rotation=45)
    plt.ylim(0, 105)  # Set y-axis limit to accommodate labels
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'parameter_accuracy_by_tool.png'), dpi=300)
    plt.close()


def generate_summary_table(results: List[Dict[str, Any]], output_dir: str) -> None:
    """Generate a summary table of results in markdown format."""
    if not results:
        return
    
    # Create markdown table
    markdown = "# Checkpoint Evaluation Summary\n\n"
    markdown += "| Checkpoint | Step | Tool Call Accuracy | Parameter Accuracy | Total Tool Calls |\n"
    markdown += "|------------|------|-------------------|-------------------|------------------|\n"
    
    for result in results:
        checkpoint = result["checkpoint"]
        step = extract_step_number(checkpoint)
        metrics = result["metrics"]
        
        tool_accuracy = metrics["tool_call_accuracy"] * 100
        param_accuracy = metrics["parameter_accuracy"] * 100
        total_calls = metrics["total_tool_calls"]
        
        checkpoint_name = os.path.basename(checkpoint)
        markdown += f"| {checkpoint_name} | {step} | {tool_accuracy:.2f}% | {param_accuracy:.2f}% | {total_calls} |\n"
    
    # Write to file
    with open(os.path.join(output_dir, 'summary_table.md'), 'w', encoding='utf-8') as f:
        f.write(markdown)


def main():
    """Main function to run the visualization."""
    parser = argparse.ArgumentParser(description="Visualize checkpoint evaluation results")
    
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to the evaluation results JSON file")
    parser.add_argument("--output_dir", type=str, default="evaluation_visualizations",
                        help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_file)
    
    if not results:
        print("No results found in the file.")
        return
    
    # Generate visualizations
    plot_accuracy_over_steps(results, args.output_dir)
    plot_tool_usage(results, args.output_dir)
    plot_parameter_accuracy_by_tool(results, args.output_dir)
    generate_summary_table(results, args.output_dir)
    
    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main() 