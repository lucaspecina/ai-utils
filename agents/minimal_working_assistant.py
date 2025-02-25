#!/usr/bin/env python3
"""
Minimal Working Assistant - No dependency issues, just works!
"""
import os
import sys
import time
import json
import requests
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
import threading
from datetime import datetime
import logging

# Configuration
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.2:latest"  # Will be auto-detected

# Set up logging to file instead of console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='assistant.log',  # Log to file instead of console
    filemode='a'
)
logger = logging.getLogger(__name__)

console = Console()

def check_ollama():
    """Check Ollama connection and find available models."""
    console.print("[bold blue]Checking Ollama connection...[/bold blue]")
    
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        if response.status_code != 200:
            console.print(f"[bold red]❌ Error: Ollama server returned {response.status_code}[/bold red]")
            console.print("Make sure Ollama is running with: ollama serve")
            sys.exit(1)
            
        # Find available models
        data = response.json()
        if "models" not in data:
            console.print("[bold red]❌ No models available in Ollama[/bold red]")
            sys.exit(1)
            
        models = [model["name"] for model in data["models"]]
        console.print(f"[bold green]✓ Connected to Ollama successfully[/bold green]")
        console.print(f"[green]Available models: {', '.join(models)}[/green]")
        
        # Find a suitable model (any llama variant)
        llama_models = [m for m in models if "llama" in m.lower()]
        
        if not llama_models:
            console.print("[bold red]❌ No Llama models found[/bold red]")
            sys.exit(1)
            
        # Use the first Llama model found
        selected_model = llama_models[0]
        console.print(f"[bold green]✓ Using model: {selected_model}[/bold green]")
        return selected_model
        
    except Exception as e:
        console.print(f"[bold red]❌ Error connecting to Ollama: {str(e)}[/bold red]")
        console.print("Make sure Ollama is running with: ollama serve")
        sys.exit(1)

class AppMonitor:
    """Simple app monitoring."""
    
    def __init__(self):
        self.recent_apps = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start(self):
        """Start app monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop app monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_loop(self):
        """Monitor active applications."""
        while self.monitoring:
            try:
                # Get processes with high CPU usage
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                    try:
                        # Force a CPU update
                        proc.cpu_percent()
                        processes.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        # Process disappeared or can't be accessed - this is normal, just skip it
                        continue
                
                # Wait for CPU measurement to be meaningful
                time.sleep(1)
                
                # Sort by CPU usage
                processes = sorted(
                    [p for p in processes if p.is_running()],  # Only include still-running processes
                    key=lambda p: p.cpu_percent(), 
                    reverse=True
                )
                
                # Take the top active processes
                active_apps = []
                for proc in processes[:5]:
                    try:
                        if proc.is_running() and proc.cpu_percent() > 0 and proc.name() not in ['python', 'Python']:
                            active_apps.append(proc.name())
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        # Process disappeared since we checked - skip it
                        continue
                
                # Store with timestamp (only if we found some)
                if active_apps:
                    timestamp = datetime.now().isoformat()
                    for app in active_apps[:3]:  # Only use top 3
                        self.recent_apps.append({
                            "app": app,
                            "timestamp": timestamp
                        })
                    
                    # Keep only the most recent entries
                    if len(self.recent_apps) > 10:
                        self.recent_apps = self.recent_apps[-10:]
                
            except Exception as e:
                # Log to file instead of console
                logger.error(f"App monitoring error: {e}")
                
            # Sleep between checks
            time.sleep(5)
    
    def get_recent_apps(self, limit=3):
        """Get list of recent applications."""
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

def main():
    """Run the minimal working assistant."""
    # Check Ollama and get model
    model_name = check_ollama()
    
    # Create app monitor
    app_monitor = AppMonitor()
    app_monitor.start()
    
    # Create conversation history
    conversation_history = []
    
    # Create interface layout
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="context", size=4),
        Layout(name="input", size=3)
    )
    
    # Set up layout
    layout["header"].update(Panel(f"Simple Assistant using {model_name}", style="bold green"))
    layout["main"].update(Panel("Type your questions below. Type 'exit' to quit.", title="Conversation"))
    layout["input"].update(Panel("", title="Your message"))
    
    conversation_display = []
    
    try:
        while True:
            # Update context panel with recent apps
            recent_apps = app_monitor.get_recent_apps()
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
            
            # Build context from history
            context = ""
            if conversation_history:
                context = "Previous conversation:\n"
                for exchange in conversation_history[-3:]:  # Last 3 exchanges
                    context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n"
            
            # Add app context
            app_context = ""
            if recent_apps:
                app_context = f"\nThe user is currently using these applications: {', '.join(recent_apps)}."
                app_context += " You can refer to these apps in your response when relevant to show awareness."
            
            # Create prompt
            system_prompt = "You are a helpful AI assistant with awareness of what applications the user is running."
            prompt = f"{system_prompt}{app_context}\n\n{context}\nUser: {user_input}\nAssistant:"
            
            # Process input
            with console.status("[bold yellow]Assistant is thinking...", spinner="dots"):
                try:
                    # Send to Ollama API directly
                    response = requests.post(
                        f"{OLLAMA_HOST}/api/generate",
                        json={"model": model_name, "prompt": prompt}
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
                        
                        # Save to history
                        conversation_history.append({
                            "user": user_input,
                            "assistant": answer.strip()
                        })
                        
                        # Add to display
                        conversation_display.append(f"[bold green]Assistant:[/bold green] {answer.strip()}")
                    else:
                        error = f"Error: Ollama returned status {response.status_code}"
                        conversation_display.append(f"[bold red]Error:[/bold red] {error}")
                        
                except Exception as e:
                    error = f"Error: {str(e)}"
                    conversation_display.append(f"[bold red]Error:[/bold red] {error}")
            
            # Update conversation panel
            conversation_text = "\n\n".join(conversation_display[-10:])
            layout["main"].update(Panel(conversation_text, title="Conversation"))
            
    finally:
        app_monitor.stop()
        console.print("[bold green]Assistant stopped. Goodbye![/bold green]")

if __name__ == "__main__":
    main() 