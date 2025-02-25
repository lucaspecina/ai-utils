# Tool-Using Fine-tuning

This directory contains scripts for fine-tuning language models to effectively use tools.

## Overview

The `tool_using.py` script provides a complete pipeline for fine-tuning a base language model to use tools effectively. It includes:

1. Synthetic data generation for tool-using scenarios
2. Data formatting and preprocessing
3. Model fine-tuning with LoRA (Low-Rank Adaptation)
4. Evaluation and model saving

## Requirements

Install the required dependencies:

```bash
pip install torch transformers datasets peft tqdm numpy
```

For efficient training with quantization, also install:

```bash
pip install bitsandbytes accelerate
```

## Usage

### Basic Usage

To generate a synthetic dataset and fine-tune a model:

```bash
python tool_using.py --model_name mistralai/Mistral-7B-v0.1 --output_dir output/tool_using_model
```

### Generate Dataset Only

To only generate the synthetic dataset without training:

```bash
python tool_using.py --generate_only --num_examples 2000
```

### Custom Training

Customize the training process with various parameters:

```bash
python tool_using.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --data_path custom_data/my_tool_dataset.json \
  --output_dir models/tool_using_llama \
  --num_examples 5000 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5
```

### Testing with Base Model

To test the base model's tool-using capabilities before training:

```bash
python tool_using.py --model_name mistralai/Mistral-7B-v0.1 --generate_only --base_model_inference
```

### Saving Checkpoints During Training

To save checkpoints at regular intervals during training:

```bash
python tool_using.py \
  --model_name mistralai/Mistral-7B-v0.1 \
  --output_dir output/tool_using_model \
  --save_strategy steps \
  --save_steps 50 \
  --eval_steps 50 \
  --save_total_limit 10
```

### Evaluating Checkpoints

To evaluate the model at different checkpoints:

```bash
python evaluate_checkpoints.py \
  --checkpoint_dir output/tool_using_model \
  --data_path data/tool_using_dataset.json \
  --num_examples 20 \
  --output_file checkpoint_evaluation.json
```

## Parameters

### Main Training Script

- `--model_name`: Base model to fine-tune (default: "mistralai/Mistral-7B-v0.1")
- `--data_path`: Path to the dataset file (default: "data/tool_using_dataset.json")
- `--output_dir`: Directory to save the fine-tuned model (default: "output/tool_using_model")
- `--num_examples`: Number of examples to generate if dataset doesn't exist (default: 1000)
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--per_device_train_batch_size`: Batch size per device during training (default: 4)
- `--gradient_accumulation_steps`: Number of gradient accumulation steps (default: 4)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--save_strategy`: When to save checkpoints (steps, epoch, or no) (default: "steps")
- `--save_steps`: Save checkpoint every X steps (default: 100)
- `--eval_steps`: Evaluate every X steps (default: 100)
- `--save_total_limit`: Maximum number of checkpoints to keep (default: 5)
- `--seed`: Random seed (default: 42)
- `--generate_only`: Only generate the dataset without training
- `--base_model_inference`: Run inference with the base model before training

### Checkpoint Evaluation Script

- `--checkpoint_dir`: Directory containing model checkpoints
- `--data_path`: Path to the dataset file
- `--base_model_name`: Name of the base model (if not specified in PEFT config)
- `--num_examples`: Number of examples to evaluate (default: 10)
- `--output_file`: Path to save evaluation results (default: "checkpoint_evaluation.json")
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.1)

## Data Format

The synthetic data follows this format:

```json
{
  "messages": [
    {"role": "user", "content": "User query"},
    {
      "role": "assistant", 
      "content": "Assistant response",
      "tool_calls": [{"name": "tool_name", "parameters": {...}}]
    },
    {"role": "tool", "content": "Tool response", "name": "tool_name"},
    {"role": "assistant", "content": "Final assistant response"}
  ],
  "tools": [...]
}
```

## Customizing Tools

You can define custom tools by modifying the `EXAMPLE_TOOLS` list in the script or by creating your own tool definitions following the `Tool` and `ToolParameter` class structure.

## Fine-tuning Approach

The script uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA to efficiently adapt the base model for tool-using capabilities. This approach:

1. Keeps most of the base model parameters frozen
2. Only trains a small set of adapter parameters
3. Significantly reduces memory requirements and training time
4. Produces a model that can effectively understand tool descriptions and generate appropriate tool calls

## Output Format

The fine-tuned model is trained to generate responses in this format:

```
<|assistant|>
I'll help you with that.

<tool_call>
name: tool_name
parameters: {
  "param1": "value1",
  "param2": "value2"
}
</tool_call>
```

## Checkpoint Evaluation

The checkpoint evaluation script allows you to:

1. Test the model at different stages of training
2. Compare tool-using capabilities across checkpoints
3. Measure tool call accuracy and parameter accuracy
4. Generate detailed reports for each checkpoint

The evaluation results are saved as a JSON file with metrics for each checkpoint. 