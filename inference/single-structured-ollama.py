"""
A simple CLI tool for getting structured responses from Ollama models.
Usage: python inference-single-structured-ollama.py --format [json|kv|list|table] "Your query here"

Examples:
    # Get JSON response
    python inference/single-structured-ollama.py --format json "Tell me about Python"

    # Get list format with specific model
    python inference/single-structured-ollama.py --format list --model codellama "List top 5 programming languages"

    # Get table format with custom temperature
    python inference/single-structured-ollama.py --format table --temperature 0.2 "Compare Python vs JavaScript"
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import argparse
import json
from typing import Any, Dict

# System prompts for different formats
FORMAT_PROMPTS = {
    'json': """You are an AI that only responds in valid JSON format.
Example: {"key": "value", "items": ["a", "b"]}
Ensure the response is always a valid JSON object.""",

    'kv': """You are an AI that responds in key-value pair format.
Format each line as 'key: value'
Example:
name: example
type: key-value
category: format""",

    'list': """You are an AI that responds in list format.
Format each item with a leading dash.
Example:
- First item
- Second item
- Third item""",

    'table': """You are an AI that responds in markdown table format.
Always include a header row and alignment row.
Example:
| Header1 | Header2 |
|---------|---------|
| Value1  | Value2  |"""
}

def get_structured_response(
    query: str,
    format_type: str,
    model: str = "deepseek-r1:8b",
    temperature: float = 0.1
) -> Any:
    """Get a single structured response from the AI
    
    Args:
        query: The user's question or prompt
        format_type: Type of structure ('json', 'kv', 'list', 'table')
        model: Name of the Ollama model to use
        temperature: Response randomness (0.0-1.0)
    
    Returns:
        Formatted response (dict for JSON, str for others)
    """
    # Initialize the language model
    llm = ChatOllama(
        model=model,
        temperature=temperature
    )

    # Create system message for requested format
    system_msg = SystemMessage(content=FORMAT_PROMPTS.get(format_type, FORMAT_PROMPTS['json']))
    
    # Create user message with format instruction
    user_msg = HumanMessage(content=f"Respond to this in {format_type} format: {query}")
    
    # Get response
    response = llm.invoke([system_msg, user_msg])
    
    # Parse JSON if requested
    if format_type == 'json':
        try:
            # Find JSON content between curly braces
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > 0:
                return json.loads(content[start:end])
            return {"error": "No JSON found in response"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in response"}
    
    return response.content

def main():
    """Parse arguments and get response"""
    parser = argparse.ArgumentParser(description='Get structured response from Ollama')
    parser.add_argument(
        'query',
        help='The question or prompt for the AI'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'kv', 'list', 'table'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--model',
        default='deepseek-r1:8b',
        help='Ollama model name (default: deepseek-r1:8b)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Response randomness (0.0-1.0, default: 0.1)'
    )
    
    args = parser.parse_args()
    
    try:
        response = get_structured_response(
            query=args.query,
            format_type=args.format,
            model=args.model,
            temperature=args.temperature
        )
        
        # Print response based on format
        if args.format == 'json':
            print(json.dumps(response, indent=2))
        else:
            print(response)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 