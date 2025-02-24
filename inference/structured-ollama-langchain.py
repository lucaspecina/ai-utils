"""
A CLI tool for getting structured responses using Langchain's output parsers.
Usage: python inference/structured-ollama-langchain.py --format [json|pydantic|list] "Your query here"

Examples:
    # Get JSON response
    python inference/structured-ollama-langchain.py --format json "Analyze Python programming language"

    # Get structured person info
    python inference/structured-ollama-langchain.py --format pydantic "Tell me about Alan Turing"

    # Get list of items
    python inference/structured-ollama-langchain.py --format list "List top cloud providers"
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.output_parsers import (
    PydanticOutputParser,
    CommaSeparatedListOutputParser,
    StructuredOutputParser,
    ResponseSchema
)
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import argparse
import json

class PersonInfo(BaseModel):
    """Example Pydantic model for structured person information"""
    name: str = Field(description="The person's full name")
    occupation: str = Field(description="Primary occupation or role")
    achievements: List[str] = Field(description="List of major achievements")
    impact: str = Field(description="Description of their impact or significance")
    dates: Dict[str, str] = Field(description="Key dates (birth, death, major events)")

def get_parser(format_type: str):
    """Get appropriate output parser based on format type"""
    if format_type == 'json':
        # Define response schemas for JSON output
        response_schemas = [
            ResponseSchema(name="summary", description="Brief summary of the topic"),
            ResponseSchema(name="details", description="Detailed information about the topic"),
            ResponseSchema(name="key_points", description="List of key points or facts"),
            ResponseSchema(name="references", description="Related topics or references")
        ]
        return StructuredOutputParser.from_response_schemas(response_schemas)
    
    elif format_type == 'pydantic':
        return PydanticOutputParser(pydantic_object=PersonInfo)
    
    elif format_type == 'list':
        return CommaSeparatedListOutputParser()
    
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

def get_format_instructions(format_type: str) -> str:
    """Get system instructions for different format types"""
    if format_type == 'json':
        return """Provide structured information in the following format:
- summary: Brief overview
- details: Comprehensive explanation
- key_points: Important facts or features
- references: Related topics or sources"""
    
    elif format_type == 'pydantic':
        return """Provide information about a person with:
- name: Full name
- occupation: Primary role
- achievements: List of accomplishments
- impact: Historical significance
- dates: Key dates dictionary"""
    
    elif format_type == 'list':
        return "Provide a comma-separated list of items"
    
    return ""

def get_structured_response(
    query: str,
    format_type: str,
    model: str = "deepseek-r1:8b",
    temperature: float = 0.1
) -> Any:
    """Get a structured response using Langchain output parsers
    
    Args:
        query: User's question or prompt
        format_type: Type of structure ('json', 'pydantic', 'list')
        model: Ollama model name
        temperature: Response randomness
    
    Returns:
        Parsed response in requested format
    """
    # Get appropriate parser
    parser = get_parser(format_type)
    
    # Initialize the language model
    llm = ChatOllama(
        model=model,
        temperature=temperature
    )
    
    # Create system message with format instructions
    system_msg = SystemMessage(content=f"""
You are an AI that provides structured responses.
{get_format_instructions(format_type)}
{parser.get_format_instructions()}
    """)
    
    # Create user message
    user_msg = HumanMessage(content=query)
    
    # Get response
    response = llm.invoke([system_msg, user_msg])
    
    # Parse the response
    try:
        return parser.parse(response.content)
    except Exception as e:
        return {"error": f"Failed to parse response: {str(e)}"}

def main():
    """Parse arguments and get response"""
    parser = argparse.ArgumentParser(description='Get structured response using Langchain parsers')
    parser.add_argument(
        'query',
        help='The question or prompt for the AI'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'pydantic', 'list'],
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
        if isinstance(response, (dict, list)):
            print(json.dumps(response, indent=2))
        else:
            print(response)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 