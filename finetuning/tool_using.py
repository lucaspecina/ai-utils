#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tool-Using Fine-tuning Script

This script fine-tunes a base language model to use tools effectively.
It includes data generation, preprocessing, and training functionality.
"""

import os
import json
import random
import argparse
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define tool schemas
@dataclass
class ToolParameter:
    name: str
    description: str
    type: str
    required: bool = False

@dataclass
class Tool:
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description
                    } for param in self.parameters
                },
                "required": [param.name for param in self.parameters if param.required]
            }
        }

# Example tools
EXAMPLE_TOOLS = [
    Tool(
        name="search_web",
        description="Search the web for information on a topic",
        parameters=[
            ToolParameter(name="query", description="The search query", type="string", required=True),
            ToolParameter(name="num_results", description="Number of results to return", type="integer")
        ]
    ),
    Tool(
        name="get_weather",
        description="Get current weather for a location",
        parameters=[
            ToolParameter(name="location", description="City or location name", type="string", required=True),
            ToolParameter(name="units", description="Temperature units (celsius/fahrenheit)", type="string")
        ]
    ),
    Tool(
        name="calculate",
        description="Perform a mathematical calculation",
        parameters=[
            ToolParameter(name="expression", description="Mathematical expression to evaluate", type="string", required=True)
        ]
    ),
]

class ToolUsingDataGenerator:
    """Generates synthetic data for tool-using fine-tuning."""
    
    def __init__(
        self, 
        tools: List[Tool] = None,
        num_examples: int = 1000,
        seed: int = 42
    ):
        self.tools = tools or EXAMPLE_TOOLS
        self.num_examples = num_examples
        random.seed(seed)
        np.random.seed(seed)
        
        # Templates for generating diverse queries
        self.query_templates = [
            "I need to {action} {subject}.",
            "Can you help me {action} {subject}?",
            "I'm trying to {action} {subject}.",
            "How can I {action} {subject}?",
            "{action} {subject} for me please.",
            "I want to {action} {subject}.",
            "Would you be able to {action} {subject}?",
            "I'd like to {action} {subject}.",
        ]
        
        # Actions and subjects for different tools
        self.tool_contexts = {
            "search_web": {
                "actions": ["find information about", "search for", "look up", "research", "find details on"],
                "subjects": [
                    "climate change", "artificial intelligence", "quantum computing", 
                    "renewable energy", "space exploration", "history of Rome",
                    "machine learning algorithms", "natural language processing",
                    "the French Revolution", "COVID-19 vaccines"
                ]
            },
            "get_weather": {
                "actions": ["check the weather in", "get the forecast for", "find out if it's raining in"],
                "subjects": [
                    "New York", "Tokyo", "London", "Paris", "Sydney", 
                    "San Francisco", "Berlin", "Toronto", "Singapore", "Mumbai"
                ]
            },
            "calculate": {
                "actions": ["calculate", "compute", "find the result of"],
                "subjects": [
                    "24 * 365", "the square root of 144", "15% of 200",
                    "5^3 + 2", "(10 + 5) * 3", "sin(30 degrees)",
                    "log base 10 of 100", "the derivative of x^2"
                ]
            }
        }
        
    def generate_tool_call(self, tool: Tool) -> Dict[str, Any]:
        """Generate a realistic tool call for a given tool."""
        tool_call = {"name": tool.name, "parameters": {}}
        
        # Fill required parameters
        for param in tool.parameters:
            if param.name == "query" and tool.name == "search_web":
                # For search queries, use the subject directly
                contexts = self.tool_contexts[tool.name]
                subject = random.choice(contexts["subjects"])
                tool_call["parameters"][param.name] = subject
            elif param.name == "location" and tool.name == "get_weather":
                # For weather, use location from subjects
                contexts = self.tool_contexts[tool.name]
                location = random.choice(contexts["subjects"])
                tool_call["parameters"][param.name] = location
            elif param.name == "expression" and tool.name == "calculate":
                # For calculator, use the expression from subjects
                contexts = self.tool_contexts[tool.name]
                expression = random.choice(contexts["subjects"])
                tool_call["parameters"][param.name] = expression
            elif param.required:
                # Generic handling for other required parameters
                if param.type == "string":
                    tool_call["parameters"][param.name] = f"sample_{param.name}"
                elif param.type == "integer":
                    tool_call["parameters"][param.name] = random.randint(1, 10)
                elif param.type == "boolean":
                    tool_call["parameters"][param.name] = random.choice([True, False])
            elif random.random() < 0.3:  # 30% chance to include optional parameters
                if param.type == "string":
                    tool_call["parameters"][param.name] = f"optional_{param.name}"
                elif param.type == "integer":
                    tool_call["parameters"][param.name] = random.randint(1, 5)
                elif param.type == "boolean":
                    tool_call["parameters"][param.name] = random.choice([True, False])
                    
        return tool_call
    
    def generate_tool_response(self, tool_call: Dict[str, Any]) -> str:
        """Generate a realistic response for a tool call."""
        tool_name = tool_call["name"]
        params = tool_call["parameters"]
        
        if tool_name == "search_web":
            query = params.get("query", "")
            return f"Here are the search results for '{query}':\n\n" + \
                   "\n".join([f"{i+1}. {query} - {random.choice(['article', 'research paper', 'blog post', 'news'])} " + 
                             f"from {random.choice(['example.com', 'research.org', 'news.com', 'blog.io'])}"
                             for i in range(min(3, params.get("num_results", 3)))])
        
        elif tool_name == "get_weather":
            location = params.get("location", "")
            temp = random.randint(5, 35)
            conditions = random.choice(["sunny", "cloudy", "rainy", "snowy", "partly cloudy"])
            units = params.get("units", "celsius")
            temp_formatted = f"{temp}°C" if units == "celsius" else f"{temp * 9/5 + 32}°F"
            return f"Current weather in {location}: {temp_formatted}, {conditions}. " + \
                   f"Humidity: {random.randint(30, 90)}%, Wind: {random.randint(0, 30)} km/h"
        
        elif tool_name == "calculate":
            expression = params.get("expression", "")
            # Simulate calculation result (not actually evaluating for safety)
            return f"The result of {expression} is {random.randint(1, 1000)}"
        
        return "Tool execution completed successfully."
    
    def generate_example(self) -> Dict[str, Any]:
        """Generate a single training example."""
        # Select a random tool
        tool = random.choice(self.tools)
        
        # Generate user query
        contexts = self.tool_contexts.get(tool.name, {"actions": ["use"], "subjects": ["this tool"]})
        action = random.choice(contexts["actions"])
        subject = random.choice(contexts["subjects"])
        template = random.choice(self.query_templates)
        user_query = template.format(action=action, subject=subject)
        
        # Generate assistant response with tool call
        tool_call = self.generate_tool_call(tool)
        tool_response = self.generate_tool_response(tool_call)
        
        # Format the assistant's response
        assistant_thinking = f"I need to use the {tool.name} tool to help with this request."
        assistant_response = f"I'll help you {action} {subject}. Let me look that up for you."
        
        # Create the full conversation
        conversation = {
            "messages": [
                {"role": "user", "content": user_query},
                {
                    "role": "assistant", 
                    "content": assistant_response,
                    "tool_calls": [tool_call]
                },
                {"role": "tool", "content": tool_response, "name": tool.name},
                {
                    "role": "assistant", 
                    "content": f"Based on the information I found: {tool_response}"
                }
            ],
            "tools": [t.to_dict() for t in self.tools]
        }
        
        return conversation
    
    def generate_dataset(self) -> List[Dict[str, Any]]:
        """Generate the full dataset."""
        logger.info(f"Generating {self.num_examples} synthetic examples...")
        examples = []
        for _ in tqdm.tqdm(range(self.num_examples)):
            examples.append(self.generate_example())
        return examples
    
    def save_dataset(self, output_path: str) -> None:
        """Generate and save the dataset to a file."""
        examples = self.generate_dataset()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2)
        
        logger.info(f"Dataset saved to {output_path}")


class ToolUsingFormatter:
    """Formats conversations for tool-using fine-tuning."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.system_prompt = (
            "You are a helpful AI assistant that can use tools to answer user questions. "
            "When a tool is needed, you should call it with the appropriate parameters. "
            "Always be helpful, accurate, and concise."
        )
    
    def format_conversation(self, example: Dict[str, Any]) -> str:
        """Format a conversation for training."""
        messages = example["messages"]
        tools = example.get("tools", [])
        
        formatted_text = f"<|system|>\n{self.system_prompt}\n"
        
        if tools:
            formatted_text += "\nYou have access to the following tools:\n"
            for tool in tools:
                formatted_text += f"- {tool['name']}: {tool['description']}\n"
        
        formatted_text += "\n"
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                formatted_text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted_text += f"<|assistant|>\n"
                
                # Handle tool calls if present
                if "tool_calls" in message and message["tool_calls"]:
                    tool_call = message["tool_calls"][0]  # Assuming one tool call per message for simplicity
                    formatted_text += f"{content}\n\n<tool_call>\n"
                    formatted_text += f"name: {tool_call['name']}\n"
                    formatted_text += f"parameters: {json.dumps(tool_call['parameters'], indent=2)}\n"
                    formatted_text += "</tool_call>\n"
                else:
                    formatted_text += f"{content}\n"
            elif role == "tool":
                formatted_text += f"<|tool|>\n{content}\n"
        
        formatted_text += "<|end|>"
        return formatted_text
    
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process a batch of examples."""
        formatted_texts = [self.format_conversation(ex) for ex in examples["conversations"]]
        
        # Tokenize the formatted conversations
        tokenized = self.tokenizer(
            formatted_texts,
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        # Prepare the labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized


def prepare_dataset(data_path: str) -> Dataset:
    """Load and prepare the dataset for training."""
    # Load the dataset
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # Generate synthetic data if file doesn't exist
        generator = ToolUsingDataGenerator()
        data = generator.generate_dataset()
        
        # Save the generated data
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_dict({"conversations": data})
    return dataset


def train_model(
    model_name: str,
    dataset: Dataset,
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_4bit: bool = True,
    use_nested_quant: bool = False,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    use_gradient_checkpointing: bool = True,
    logging_steps: int = 10,
    save_strategy: str = "epoch",
    seed: int = 42
) -> None:
    """Fine-tune a model for tool-using capabilities."""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization for efficient training
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    ) if use_4bit else None
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training if using quantization
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    logger.info("Configuring LoRA")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    
    # Enable gradient checkpointing for memory efficiency
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Format the dataset
    formatter = ToolUsingFormatter(tokenizer)
    
    # Process the dataset
    logger.info("Processing dataset")
    processed_dataset = dataset.map(
        lambda examples: {"conversations": examples["conversations"]},
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Apply the preprocessing function
    tokenized_dataset = processed_dataset.map(
        formatter.preprocess_function,
        batched=True,
        remove_columns=["conversations"]
    )
    
    # Split dataset into train and validation
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=seed)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        evaluation_strategy="epoch",
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        push_to_hub=False,
        load_best_model_at_end=True,
        seed=seed
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator
    )
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    # Save the fine-tuned model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    """Main function to run the tool-using fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune a model for tool-using capabilities")
    
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1", 
                        help="Base model to fine-tune")
    parser.add_argument("--data_path", type=str, default="data/tool_using_dataset.json",
                        help="Path to the dataset file (will be generated if it doesn't exist)")
    parser.add_argument("--output_dir", type=str, default="output/tool_using_model",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--num_examples", type=int, default=1000,
                        help="Number of examples to generate if dataset doesn't exist")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--generate_only", action="store_true",
                        help="Only generate the dataset without training")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate or load dataset
    if not os.path.exists(args.data_path) or args.generate_only:
        logger.info(f"Generating synthetic dataset with {args.num_examples} examples")
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        generator = ToolUsingDataGenerator(num_examples=args.num_examples, seed=args.seed)
        generator.save_dataset(args.data_path)
        
        if args.generate_only:
            logger.info(f"Dataset generated at {args.data_path}. Exiting as --generate_only was specified.")
            return
    
    # Prepare dataset
    dataset = prepare_dataset(args.data_path)
    
    # Train the model
    train_model(
        model_name=args.model_name,
        dataset=dataset,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed
    )
    
    logger.info(f"Fine-tuning completed. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
