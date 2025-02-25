#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ollama Adapter for Tool-Using Models

This module provides functions to interface with Ollama for tool-using models.
"""

import json
import requests
import logging
import time
import re
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api"


def format_prompt_for_ollama(user_query: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Format a prompt for Ollama."""
    default_system_prompt = (
        "You are a helpful AI assistant that can use tools to answer user questions. "
        "When a tool is needed, you should call it with the appropriate parameters. "
        "Always be helpful, accurate, and concise.\n\n"
        "You have access to the following tools:\n"
        "- search_web: Search the web for information on a topic\n"
        "- get_weather: Get current weather for a location\n"
        "- calculate: Perform a mathematical calculation\n\n"
        "When you need to use a tool, format your response like this:\n\n"
        "<tool_call>\n"
        "name: tool_name\n"
        "parameters: {\n"
        '  "param1": "value1",\n'
        '  "param2": "value2"\n'
        "}\n"
        "</tool_call>\n"
    )
    
    system_prompt = system_prompt or default_system_prompt
    
    # Format messages for Ollama
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    return {
        "model": "llama3.1:latest",
        "messages": messages,
        "stream": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1024
    }


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
            logger.warning(f"Could not parse parameters for tool call: {tool_name}")
            continue
    
    return tool_calls


def generate_response_with_ollama(user_query: str, model: str = "llama3.1:latest", 
                                  system_prompt: Optional[str] = None,
                                  api_url: str = OLLAMA_API_URL,
                                  temperature: float = 0.7,
                                  max_tokens: int = 1024) -> str:
    """Generate a response using Ollama."""
    # Format the prompt
    request_data = format_prompt_for_ollama(user_query, system_prompt)
    
    # Override defaults if provided
    request_data["model"] = model
    request_data["temperature"] = temperature
    request_data["max_tokens"] = max_tokens
    
    # Send request to Ollama
    try:
        response = requests.post(f"{api_url}/chat/completions", json=request_data)
        response.raise_for_status()
        
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"Unexpected response format from Ollama: {result}")
            return "Error: Unexpected response format from Ollama."
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with Ollama: {e}")
        return f"Error communicating with Ollama: {e}"


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


def run_ollama_tool_using_test(query: str, model: str = "llama3.1:latest") -> Dict[str, Any]:
    """Run a complete tool-using test with Ollama, including tool execution."""
    # Step 1: Generate initial response
    logger.info(f"Generating response for query: {query}")
    response = generate_response_with_ollama(query, model=model)
    
    # Step 2: Extract tool calls
    tool_calls = extract_tool_calls(response)
    
    result = {
        "query": query,
        "initial_response": response,
        "tool_calls": tool_calls,
        "tool_responses": [],
        "final_response": None
    }
    
    # Step 3: Execute tools and generate final response if tool calls present
    if tool_calls:
        for tool_call in tool_calls:
            # Execute tool
            tool_response = simulate_tool_execution(tool_call)
            result["tool_responses"].append({
                "tool_call": tool_call,
                "response": tool_response
            })
            
            # Generate follow-up response with tool results
            follow_up_query = f"{query}\n\nTool response: {tool_response}"
            follow_up_response = generate_response_with_ollama(follow_up_query, model=model)
            result["final_response"] = follow_up_response
    
    return result


def test_ollama_setup():
    """Test if Ollama is running and the specified model is available."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            
            logger.info(f"Ollama is running. Available models: {available_models}")
            
            if "llama3.1:latest" in available_models:
                logger.info("llama3.1:latest is available!")
                return True
            else:
                logger.warning("llama3.1:latest not found. Please pull it with 'ollama pull llama3.1:latest'")
                return False
        else:
            logger.error(f"Error checking Ollama models: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Could not connect to Ollama: {e}")
        logger.error("Make sure Ollama is running on localhost:11434")
        return False


if __name__ == "__main__":
    # Simple test
    if test_ollama_setup():
        test_query = "What's the weather like in Paris?"
        result = run_ollama_tool_using_test(test_query)
        
        print(f"Query: {result['query']}")
        print(f"Initial response: {result['initial_response']}")
        
        if result['tool_calls']:
            print(f"Tool calls detected: {len(result['tool_calls'])}")
            for i, tool_call in enumerate(result['tool_calls']):
                print(f"Tool call {i+1}:")
                print(f"  Name: {tool_call['name']}")
                print(f"  Parameters: {json.dumps(tool_call['parameters'], indent=2)}")
                
            for i, tool_response in enumerate(result['tool_responses']):
                print(f"Tool response {i+1}: {tool_response['response']}")
                
            if result['final_response']:
                print(f"Final response: {result['final_response']}")
        else:
            print("No tool calls detected in the response") 