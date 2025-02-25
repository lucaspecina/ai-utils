#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete Tool-Using Fine-tuning Workflow

This script runs the complete workflow:
1. Generate synthetic data
2. Test the base model
3. Fine-tune the model with checkpoints
4. Evaluate all checkpoints
5. Visualize the results
"""

import os
import argparse
import subprocess
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_command(command: str) -> int:
    """Run a shell command and log the output."""
    logger.info(f"Running command: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Stream the output
    for line in process.stdout:
        logger.info(line.strip())
    
    # Wait for the process to complete
    process.wait()
    
    return process.returncode


def main():
    """Main function to run the complete workflow."""
    parser = argparse.ArgumentParser(description="Run the complete tool-using fine-tuning workflow")
    
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1",
                        help="Base model to fine-tune")
    parser.add_argument("--num_examples", type=int, default=500,
                        help="Number of examples to generate")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save all outputs (default: timestamped directory)")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval_examples", type=int, default=20,
                        help="Number of examples to use for evaluation")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip the training step (for testing other parts of the workflow)")
    
    args = parser.parse_args()
    
    # Create timestamped output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output/tool_using_{timestamp}"
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    data_dir = os.path.join(args.output_dir, "data")
    model_dir = os.path.join(args.output_dir, "model")
    eval_dir = os.path.join(args.output_dir, "evaluation")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Define file paths
    data_path = os.path.join(data_dir, "tool_using_dataset.json")
    eval_results_path = os.path.join(eval_dir, "checkpoint_evaluation.json")
    
    # Step 1: Generate synthetic data and test base model
    logger.info("Step 1: Generating synthetic data and testing base model")
    data_gen_cmd = (
        f"python tool_using.py --generate_only --base_model_inference "
        f"--model_name {args.model_name} --num_examples {args.num_examples} "
        f"--data_path {data_path}"
    )
    if run_command(data_gen_cmd) != 0:
        logger.error("Data generation failed. Exiting.")
        return
    
    # Step 2: Fine-tune the model with checkpoints
    if not args.skip_training:
        logger.info("Step 2: Fine-tuning the model with checkpoints")
        train_cmd = (
            f"python tool_using.py --model_name {args.model_name} "
            f"--data_path {data_path} --output_dir {model_dir} "
            f"--num_train_epochs {args.num_train_epochs} "
            f"--save_strategy steps --save_steps {args.save_steps} "
            f"--eval_steps {args.save_steps} --save_total_limit 10"
        )
        if run_command(train_cmd) != 0:
            logger.error("Training failed. Exiting.")
            return
    else:
        logger.info("Skipping training as requested.")
    
    # Step 3: Evaluate all checkpoints
    logger.info("Step 3: Evaluating all checkpoints")
    eval_cmd = (
        f"python evaluate_checkpoints.py --checkpoint_dir {model_dir} "
        f"--data_path {data_path} --num_examples {args.eval_examples} "
        f"--output_file {eval_results_path} --base_model_name {args.model_name}"
    )
    if run_command(eval_cmd) != 0:
        logger.error("Evaluation failed. Exiting.")
        return
    
    # Step 4: Visualize the results
    logger.info("Step 4: Visualizing the results")
    viz_cmd = (
        f"python visualize_results.py --results_file {eval_results_path} "
        f"--output_dir {eval_dir}"
    )
    if run_command(viz_cmd) != 0:
        logger.error("Visualization failed. Exiting.")
        return
    
    logger.info(f"Complete workflow finished successfully. All outputs saved to {args.output_dir}")
    logger.info(f"- Data: {data_path}")
    logger.info(f"- Model: {model_dir}")
    logger.info(f"- Evaluation: {eval_dir}")


if __name__ == "__main__":
    main() 