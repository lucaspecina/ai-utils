# Using Ollama with Llama 3.1 for Tool-Using Tasks

This guide explains how to use Ollama with Llama 3.1 for tool-using capabilities.

## Overview

The provided scripts allow you to:

1. Test Llama 3.1 on tool-using tasks through Ollama
2. Evaluate the model's tool-using capabilities
3. Generate synthetic data for testing
4. Compare Llama 3.1's performance with fine-tuned models

## Requirements

1. [Ollama](https://ollama.com/download) installed
2. Llama 3.1 model pulled in Ollama
3. Python 3.8+ environment with dependencies installed

## Setup

### 1. Install Ollama

Follow the installation instructions at [ollama.com/download](https://ollama.com/download)

### 2. Pull Llama 3.1

```bash
ollama pull llama3.1:latest
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Testing Llama 3.1 with Tools

You can quickly test Llama 3.1's tool-using capabilities with the `ollama_adapter.py` script:

```bash
python ollama_adapter.py
```

This will run a simple test with the query "What's the weather like in Paris?" and check if the model correctly generates a tool call.

## Evaluating Llama 3.1 on Tool-Using Tasks

To evaluate Llama 3.1 on a dataset of tool-using tasks:

```bash
# First, generate a synthetic dataset (if you don't have one already)
python tool_using.py --generate_only --num_examples 100

# Then evaluate Llama 3.1 using Ollama
python evaluate_ollama.py --data_path data/tool_using_dataset.json --num_examples 20
```

The evaluation results will be saved to `ollama_evaluation.json` by default.

## Advanced Usage

### Using Different Models

You can specify different models available in your Ollama installation:

```bash
python evaluate_ollama.py --data_path data/tool_using_dataset.json --model llama3.1:8b
```

### Customizing System Prompt

The default system prompt in `ollama_adapter.py` can be modified to customize the tool descriptions or format instructions.

### Comparing with Fine-tuned Models

To compare Llama 3.1 in Ollama with a fine-tuned model:

1. Generate a dataset and evaluate Llama 3.1:
```bash
python tool_using.py --generate_only --num_examples 100
python evaluate_ollama.py --data_path data/tool_using_dataset.json --num_examples 20 --output_file ollama_results.json
```

2. Fine-tune a model using the original workflow:
```bash
python tool_using.py --model_name mistralai/Mistral-7B-v0.1 --data_path data/tool_using_dataset.json
```

3. Evaluate the fine-tuned model:
```bash
python evaluate_checkpoints.py --checkpoint_dir output/tool_using_model --data_path data/tool_using_dataset.json --num_examples 20 --output_file finetuned_results.json
```

4. Compare the results:
```bash
# Manually compare the metrics in both JSON files
```

## Potential Fine-tuning with Ollama

While the current workflow uses PEFT fine-tuning with Hugging Face Transformers, you could potentially fine-tune models through Ollama as well. This would require:

1. Setting up a local fine-tuning environment that works with Ollama
2. Converting the synthetic dataset into a format suitable for that environment
3. Converting the fine-tuned model back to a format Ollama can use

This is more advanced and not directly supported by the scripts in this repository.

## Files Overview

- `ollama_adapter.py`: Core utilities for interfacing with Ollama
- `evaluate_ollama.py`: Script for evaluating Llama 3.1 on tool-using tasks
- `tool_using.py`: Original script for data generation and training (with Transformers)
- `OLLAMA_README.md`: This documentation file

## Limitations

- Ollama's response format may differ slightly from the formats used in fine-tuning
- Local models through Ollama may have different performance characteristics than cloud-hosted variants
- Evaluating large numbers of examples may be time-consuming on consumer hardware 