#!/usr/bin/env python3
"""
Simplified AI Personal Assistant with App Awareness
"""

import os
import sys
import time
import datetime
import threading
import json
import logging
from typing import Dict, List
import requests  # Using requests instead of ollama package

# Core dependencies
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout

# Configure logging (file only)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("assistant.log")]
)
logger = logging.getLogger("assistant")

class AppMonitor:
    """Simple app monitoring without excessive dependencies"""
    
    def __init__(self):
        self.recent_apps = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start(self):
        """Start app monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop app monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_loop(self):
        """Monitor active applications"""
        while self.monitoring:
            try:
                # Simple approach - just get process name with highest CPU usage
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                    try:
                        process_info = proc.info
                        if process_info['cpu_percent'] > 0:
                            processes.append(process_info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # Sort by CPU usage
                processes = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)
                
                # Take the top 3 processes
                active_apps = []
                for p in processes[:3]:
                    if p['name'] not in active_apps and p['name'] != 'python' and p['name'] != 'Python':
                        active_apps.append(p['name'])
                
                # Store with timestamp
                if active_apps:
                    timestamp = datetime.datetime.now().isoformat()
                    for app in active_apps:
                        self.recent_apps.append({
                            "app": app,
                            "timestamp": timestamp
                        })
                    
                    # Keep only the most recent 10 entries
                    if len(self.recent_apps) > 10:
                        self.recent_apps = self.recent_apps[-10:]
                
            except Exception as e:
                logger.error(f"App monitoring error: {e}")
                
            # Sleep for 5 seconds
            time.sleep(5)
    
    def get_recent_apps(self, limit=3):
        """Get list of recent applications"""
        # Get unique app names from recent history, most recent first
        seen = set()
        unique_apps = []
        
        for item in reversed(self.recent_apps):
            app_name = item["app"]
            if app_name not in seen:
                seen.add(app_name)
                unique_apps.append(app_name)
                if len(unique_apps) >= limit:
                    break
                    
        return unique_apps


class SimpleAssistant:
    """Simple assistant with app awareness"""
    
    def __init__(self):
        self.conversation_history = []
        self.app_monitor = AppMonitor()
        self.model_name = "llama3.2:latest"  # Default to llama3.2:latest
        
        # Check Ollama connection - using requests instead of ollama package
        self._check_ollama()
        
        # Start app monitoring
        self.app_monitor.start()
    
    def _check_ollama(self):
        """Verify Ollama is running using direct HTTP requests."""
        try:
            # Use requests instead of ollama package
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                print("✓ Connected to Ollama server successfully")
                
                # Check models
                data = response.json()
                if "models" in data:
                    models = [model["name"] for model in data["models"]]
                    print(f"✓ Available models: {', '.join(models)}")
                    
                    # Look for any llama3 variant
                    llama_models = [m for m in models if "llama3" in m]
                    if llama_models:
                        # Use the first available llama3 variant
                        self.model_name = llama_models[0]
                        print(f"✓ Using model: {self.model_name}")
                    else:
                        print("! No llama3 variant found - please run: ollama pull llama3")
                        sys.exit(1)
                else:
                    print("! No models found - please run: ollama pull llama3")
                    sys.exit(1)
            else:
                print(f"✗ Ollama server returned error code: {response.status_code}")
                print("  Make sure Ollama is running with 'ollama serve'")
                sys.exit(1)
        except Exception as e:
            print(f"✗ Failed to connect to Ollama: {e}")
            print("  Make sure Ollama is running with 'ollama serve'")
            sys.exit(1)
    
    def stop(self):
        """Stop the assistant"""
        self.app_monitor.stop()
    
    def process_input(self, user_input):
        """Process user input and generate a response using direct API calls."""
        # Get recent apps
        recent_apps = self.app_monitor.get_recent_apps()
        
        # Build context from conversation history
        context = ""
        if self.conversation_history:
            context = "Previous conversation:\n"
            for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
                context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n"
        
        # Build app context
        app_context = ""
        if recent_apps:
            app_context = f"\nThe user is currently using these applications: {', '.join(recent_apps)}."
            app_context += " You can refer to these apps in your response when relevant to show awareness."
        
        # Create prompt
        system_prompt = "You are a helpful AI assistant with awareness of what applications the user is running."
        prompt = f"{system_prompt}{app_context}\n\n{context}\nUser: {user_input}\nAssistant:"
        
        # Send to Ollama using direct API
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model_name, "prompt": prompt}  # Use the detected model
            )
            
            if response.status_code == 200:
                # Process streaming response
                answer = ""
                for line in response.text.splitlines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                answer += data["response"]
                        except json.JSONDecodeError:
                            pass
                
                # Save to conversation history
                self.conversation_history.append({
                    "user": user_input,
                    "assistant": answer.strip()
                })
                
                return answer.strip()
            else:
                error_msg = f"Error: Ollama returned status code {response.status_code}"
                logger.error(error_msg)
                return f"I encountered an error communicating with my language model. {error_msg}"
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error: {str(e)}"


def main():
    """Run the simple assistant."""
    console = Console()
    
    # Create assistant
    with console.status("[bold green]Initializing assistant...", spinner="dots"):
        assistant = SimpleAssistant()
    
    # Create interface layout
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="context", size=4),
        Layout(name="input", size=3)
    )
    
    # Set up layout
    layout["header"].update(Panel("Simple AI Assistant with App Awareness", style="bold green"))
    layout["main"].update(Panel("Type your questions below. Type 'exit' to quit.", title="Conversation"))
    layout["input"].update(Panel("", title="Your message"))
    
    conversation_display = []
    
    # Main interaction loop
    try:
        while True:
            # Update context panel with recent apps
            recent_apps = assistant.app_monitor.get_recent_apps()
            app_text = "No applications detected" if not recent_apps else ", ".join(recent_apps)
            layout["context"].update(Panel(f"[bold]Recent Applications:[/bold]\n{app_text}", title="System Context"))
            
            # Display layout
            console.clear()
            console.print(layout)
            
            # Get user input
            user_input = console.input("[bold cyan]You > [/bold cyan]")
            
            if user_input.lower() == "exit":
                break
            
            # Add to conversation display
            conversation_display.append(f"[bold cyan]You:[/bold cyan] {user_input}")
            
            # Process input
            with console.status("[bold yellow]Assistant is thinking...", spinner="dots"):
                response = assistant.process_input(user_input)
            
            # Add response to conversation display
            conversation_display.append(f"[bold green]Assistant:[/bold green] {response}")
            
            # Update conversation panel
            conversation_text = "\n\n".join(conversation_display[-10:])
            layout["main"].update(Panel(conversation_text, title="Conversation"))
            
    finally:
        assistant.stop()
        console.print("[bold green]Assistant stopped. Goodbye![/bold green]")

if __name__ == "__main__":
    main() 