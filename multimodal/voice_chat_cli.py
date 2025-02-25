#!/usr/bin/env python3
"""
Voice Chat CLI - A multimodal interface for conversing with AI using voice.
This application uses state-of-the-art open source components:
- Whisper (local) for speech-to-text
- Ollama for AI processing (supports various open source models)
- Piper TTS for high-quality open source text-to-speech
- PyAudio for audio recording
"""

import os
import sys
import time
import argparse
import tempfile
import wave
import subprocess
import json
from typing import Optional, List, Dict, Any
from pathlib import Path

import pyaudio
import numpy as np
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress

# Load environment variables from .env file
load_dotenv()

# Initialize console for rich output
console = Console()

# Configuration
DEFAULT_MODEL = "llama3.2:latest"  # Default Ollama model
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 500  # Adjust based on your microphone sensitivity
SILENCE_DURATION = 2.0  # Seconds of silence to end recording

# Ollama API endpoint
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api")

class VoiceChatCLI:
    def __init__(
        self, 
        model: str = DEFAULT_MODEL,
        voice: str = "en_US-lessac-medium",  # Default Piper voice
        use_tts: bool = True,
        save_conversations: bool = True,
        conversation_dir: Optional[str] = None,
        whisper_model: str = "base",  # Whisper model size: tiny, base, small, medium, large
        piper_model_dir: Optional[str] = None
    ):
        self.model = model
        self.voice = voice
        self.use_tts = use_tts
        self.save_conversations = save_conversations
        self.whisper_model = whisper_model
        
        # Set Piper model directory
        if piper_model_dir:
            self.piper_model_dir = Path(piper_model_dir)
        else:
            self.piper_model_dir = Path.home() / ".local" / "share" / "piper-tts" / "voices"
        
        if conversation_dir:
            self.conversation_dir = Path(conversation_dir)
        else:
            self.conversation_dir = Path.home() / ".voice_chat_history"
            
        if self.save_conversations:
            self.conversation_dir.mkdir(exist_ok=True, parents=True)
            
        self.conversation_history = []
        self.pyaudio_instance = pyaudio.PyAudio()
        
        # Check for required dependencies
        self._check_dependencies()
        
    def _check_dependencies(self):
        """Check if required dependencies are installed."""
        missing_deps = []
        
        # Check for whisper
        try:
            import whisper
            self.whisper = whisper
        except ImportError:
            missing_deps.append("openai-whisper")
        
        # Check for Ollama
        try:
            response = requests.get(f"{OLLAMA_API_URL}/tags")
            if response.status_code != 200:
                missing_deps.append("ollama (service not running)")
        except requests.exceptions.ConnectionError:
            missing_deps.append("ollama (service not running)")
        
        # Check for Piper if TTS is enabled
        if self.use_tts:
            try:
                result = subprocess.run(["piper", "--help"], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE, 
                                       text=True)
                if result.returncode != 0:
                    missing_deps.append("piper-tts")
            except FileNotFoundError:
                missing_deps.append("piper-tts")
        
        if missing_deps:
            console.print("[bold red]Missing dependencies:[/bold red]")
            for dep in missing_deps:
                console.print(f"- {dep}")
            console.print("\n[yellow]Installation instructions:[/yellow]")
            console.print("1. For Whisper: pip install openai-whisper")
            console.print("2. For Ollama: https://ollama.com/download")
            console.print("3. For Piper TTS: https://github.com/rhasspy/piper#installation")
            sys.exit(1)
        
    def record_audio(self) -> str:
        """Record audio from microphone until silence is detected."""
        console.print("\n[bold blue]🎤 Recording... (speak now, pause to end)[/bold blue]")
        
        # Open temporary file for WAV audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
        
        # Setup audio stream
        stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        frames = []
        silent_chunks = 0
        silent_threshold = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
        
        # Start recording with progress indicator
        with Progress() as progress:
            task = progress.add_task("[cyan]Recording...", total=None)
            
            try:
                while True:
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(data)
                    
                    # Check for silence
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    if np.abs(audio_data).mean() < SILENCE_THRESHOLD:
                        silent_chunks += 1
                        if silent_chunks >= silent_threshold:
                            break
                    else:
                        silent_chunks = 0
                        
                    progress.update(task, advance=1)
            
            except KeyboardInterrupt:
                console.print("\n[yellow]Recording stopped by user[/yellow]")
            
            finally:
                stream.stop_stream()
                stream.close()
        
        # Save recorded audio to WAV file
        with wave.open(temp_wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.pyaudio_instance.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
        
        console.print("[green]Recording complete![/green]")
        return temp_wav_path
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio file using local Whisper model."""
        console.print(f"[blue]Transcribing audio with Whisper ({self.whisper_model})...[/blue]")
        
        # Load the Whisper model
        model = self.whisper.load_model(self.whisper_model)
        
        # Transcribe the audio
        result = model.transcribe(audio_path)
        
        os.unlink(audio_path)  # Clean up the temporary file
        return result["text"]
    
    def get_ai_response(self, user_input: str) -> str:
        """Get AI response using Ollama API."""
        console.print(f"[blue]Getting AI response from {self.model}...[/blue]")
        
        # Format messages for Ollama
        messages = []
        for msg in self.conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add user message to conversation history and messages
        self.conversation_history.append({"role": "user", "content": user_input})
        messages.append({"role": "user", "content": user_input})
        
        # Get response from Ollama
        response = requests.post(
            f"{OLLAMA_API_URL}/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False
            }
        )
        
        if response.status_code != 200:
            console.print(f"[red]Error from Ollama API: {response.status_code}[/red]")
            console.print(response.text)
            return "I'm sorry, I encountered an error while processing your request."
        
        ai_response = response.json()["message"]["content"]
        
        # Add AI response to conversation history
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        
        return ai_response
    
    def text_to_speech(self, text: str) -> None:
        """Convert text to speech using Piper TTS."""
        console.print("[blue]Converting to speech with Piper TTS...[/blue]")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
        
        try:
            # Run Piper TTS
            process = subprocess.Popen(
                ["piper", "--model", f"{self.voice}", "--output_file", temp_wav_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send text to Piper
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode != 0:
                console.print(f"[red]Error in text-to-speech: {stderr}[/red]")
                return
            
            # Play the audio using system default player
            if sys.platform == "darwin":  # macOS
                os.system(f"afplay {temp_wav_path}")
            elif sys.platform == "linux":
                os.system(f"aplay {temp_wav_path}")
            elif sys.platform == "win32":
                os.system(f"start {temp_wav_path}")
            
        except Exception as e:
            console.print(f"[red]Error in text-to-speech: {str(e)}[/red]")
        
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_wav_path)
            except:
                pass
    
    def save_conversation(self) -> None:
        """Save the conversation history to a JSON file."""
        if not self.save_conversations or not self.conversation_history:
            return
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = self.conversation_dir / f"conversation-{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        
        console.print(f"[green]Conversation saved to {filename}[/green]")
    
    def run(self) -> None:
        """Run the voice chat CLI in interactive mode."""
        console.print(Panel.fit(
            f"[bold cyan]Open Source Voice Chat CLI[/bold cyan]\n"
            f"Using [green]{self.model}[/green] with Ollama\n"
            f"Whisper [green]{self.whisper_model}[/green] for speech recognition\n"
            f"Piper voice: [green]{self.voice}[/green]\n"
            "Speak to the AI and get voice responses.\n"
            "Press Ctrl+C to exit at any time.",
            title="Welcome"
        ))
        
        try:
            # Add system message to set the tone
            self.conversation_history.append({
                "role": "system", 
                "content": "You are a helpful, friendly AI assistant. Respond concisely and clearly."
            })
            
            while True:
                # Record and transcribe user input
                audio_path = self.record_audio()
                user_input = self.transcribe_audio(audio_path)
                
                console.print(f"\n[bold green]You said:[/bold green] {user_input}")
                
                # Get AI response
                ai_response = self.get_ai_response(user_input)
                
                # Display AI response
                console.print("\n[bold purple]AI:[/bold purple]")
                console.print(Markdown(ai_response))
                
                # Convert to speech if enabled
                if self.use_tts:
                    self.text_to_speech(ai_response)
                
                console.print("\n" + "-" * 50 + "\n")
        
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Exiting Voice Chat CLI...[/bold yellow]")
        
        finally:
            self.save_conversation()
            self.pyaudio_instance.terminate()

def list_ollama_models() -> None:
    """List available Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            console.print("[bold cyan]Available Ollama Models:[/bold cyan]")
            
            for model in models:
                console.print(f"[green]Name:[/green] {model['name']}")
                console.print(f"[green]Size:[/green] {model.get('size', 'Unknown')}")
                console.print(f"[green]Modified:[/green] {model.get('modified_at', 'Unknown')}")
                console.print("-" * 50)
        else:
            console.print(f"[red]Error: {response.status_code}[/red]")
            console.print(response.text)
    except requests.exceptions.ConnectionError:
        console.print("[bold red]Error: Could not connect to Ollama API[/bold red]")
        console.print("Make sure Ollama is running (ollama serve)")

def list_piper_voices() -> None:
    """List available Piper TTS voices."""
    try:
        result = subprocess.run(
            ["piper", "--list_voices"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        if result.returncode == 0:
            console.print("[bold cyan]Available Piper TTS Voices:[/bold cyan]")
            console.print(result.stdout)
        else:
            console.print(f"[red]Error listing Piper voices: {result.stderr}[/red]")
    except FileNotFoundError:
        console.print("[bold red]Error: Piper TTS not found[/bold red]")
        console.print("Install Piper TTS: https://github.com/rhasspy/piper#installation")

def main():
    parser = argparse.ArgumentParser(description="Open Source Voice Chat CLI - Talk with AI using your voice")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--voice", type=str, default="en_US-lessac-medium", help="Piper TTS voice to use")
    parser.add_argument("--whisper-model", type=str, default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech output")
    parser.add_argument("--no-save", action="store_true", help="Don't save conversation history")
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models and exit")
    parser.add_argument("--list-voices", action="store_true", help="List available Piper TTS voices and exit")
    parser.add_argument("--conversation-dir", type=str, help="Directory to save conversation history")
    parser.add_argument("--piper-model-dir", type=str, help="Directory containing Piper voice models")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_ollama_models()
        return
    
    if args.list_voices:
        list_piper_voices()
        return
    
    chat_cli = VoiceChatCLI(
        model=args.model,
        voice=args.voice,
        use_tts=not args.no_tts,
        save_conversations=not args.no_save,
        conversation_dir=args.conversation_dir,
        whisper_model=args.whisper_model,
        piper_model_dir=args.piper_model_dir
    )
    
    chat_cli.run()

if __name__ == "__main__":
    main() 