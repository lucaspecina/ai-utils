#!/usr/bin/env python3
"""
State-of-the-Art AI Personal Assistant

This module implements a cutting-edge AI personal assistant using open source models and Ollama.
"""

import os
import sys
import time
import json
import logging
import datetime
import threading
import queue
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import required libraries
import ollama
import pyautogui
import psutil
import speech_recognition as sr
import pyttsx3
import numpy as np
from PIL import Image
import faiss
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
import tqdm.auto
import warnings
from contextlib import redirect_stdout, redirect_stderr
import io
import base64

# Silence tqdm progress bars from sentence-transformers (causing the "batches" logs)
# Add this code right after your imports section
tqdm.auto.tqdm = lambda *args, **kwargs: tqdm.auto.tqdm(*args, **kwargs, disable=True)
warnings.filterwarnings("ignore")

# Suppress certain logs
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("assistant.log"),  # Keep this for debugging
        # Remove the StreamHandler to keep logs out of the terminal
    ]
)
logger = logging.getLogger("sota_assistant")

# Create a separate console handler with higher threshold for critical messages only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)  # Only show errors in console
formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Default configuration
DEFAULT_CONFIG = {
    "monitoring": {
        "enable_system_logs": True,
        "enable_screenshots": False,
        "screenshot_interval": 300,  # seconds
        "privacy_sensitive_apps": ["password manager", "banking", "private browsing"],
    },
    "multimodal": {
        "enable_voice": True,
        "enable_vision": True,
        "voice_trigger_phrase": "hey assistant",
    },
    "memory": {
        "short_term_limit": 10,  # conversations
        "vector_db_path": "memory/vector_store",
        "user_preferences_path": "memory/user_prefs.json",
    },
    "models": {
        "text_model": "llama3",  # Ollama model for text
        "vision_model": "llava",  # Ollama model for vision
        "embedding_model": "nomic-embed-text",  # Ollama embedding model
        "voice_model": "vosk",  # Open source speech recognition
        "ollama_host": "http://localhost:11434"  # Ollama API endpoint
    }
}

# Add colorful prints for better visibility
try:
    from colorama import init, Fore, Style
    init()  # Initialize colorama
except ImportError:
    # Define fallback classes if colorama isn't installed
    class DummyColorama:
        def __init__(self):
            self.RESET = ''
            self.RED = ''
            self.GREEN = ''
            self.YELLOW = ''
            self.BLUE = ''
            self.MAGENTA = ''
            self.CYAN = ''
    
    class DummyStyle:
        def __init__(self):
            self.RESET_ALL = ''
    
    # Create dummy objects
    Fore = DummyColorama()
    Style = DummyStyle()
    print("Note: Install 'colorama' package for colored output (pip install colorama)")

class ModalityType(Enum):
    """Types of input/output modalities supported by the assistant."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MultimodalInput:
    """Container for multimodal inputs to the assistant."""
    text: Optional[str] = None
    image: Optional[Any] = None  # PIL Image or path
    audio: Optional[Any] = None  # Audio data or path
    video: Optional[Any] = None  # Video data or path
    modality_type: ModalityType = ModalityType.TEXT


@dataclass
class MultimodalOutput:
    """Container for multimodal outputs from the assistant."""
    text: Optional[str] = None
    image: Optional[Any] = None
    audio: Optional[Any] = None
    video: Optional[Any] = None
    modality_type: ModalityType = ModalityType.TEXT


class ActivityMonitor:
    """
    Monitors user computer activities with privacy considerations.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.monitoring_active = False
        self.activity_queue = queue.Queue()
        self.screenshot_thread = None
        self.app_monitor_thread = None
        
        # Privacy settings
        self.privacy_mode = False
        self.privacy_sensitive_apps = config["monitoring"]["privacy_sensitive_apps"]
        
        # Initialize storage for collected data
        os.makedirs("activity_data", exist_ok=True)
        
    def start_monitoring(self):
        """Start all monitoring activities based on configuration."""
        self.monitoring_active = True
        
        if self.config["monitoring"]["enable_system_logs"]:
            self._start_app_monitoring()
            
        if self.config["monitoring"]["enable_screenshots"]:
            self._start_screenshot_capture()
            
        logger.info("Activity monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring activities."""
        self.monitoring_active = False
        logger.info("Activity monitoring stopped")
    
    def _start_screenshot_capture(self):
        """Start the screenshot capturing thread."""
        def capture_screenshots():
            interval = self.config["monitoring"]["screenshot_interval"]
            while self.monitoring_active:
                if not self.privacy_mode:
                    try:
                        # Capture screenshot using PyAutoGUI
                        screenshot = pyautogui.screenshot()
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Save screenshot temporarily for processing
                        temp_path = f"activity_data/temp_screenshot_{timestamp}.png"
                        screenshot.save(temp_path)
                        
                        # Process screenshot with vision model
                        processed_data = self._process_screenshot(temp_path)
                        
                        # Remove temporary file after processing
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        self.activity_queue.put({
                            "type": "screenshot",
                            "timestamp": timestamp,
                            "data": processed_data
                        })
                    except Exception as e:
                        logger.error(f"Screenshot capture error: {e}")
                
                time.sleep(interval)
        
        self.screenshot_thread = threading.Thread(target=capture_screenshots)
        self.screenshot_thread.daemon = True
        self.screenshot_thread.start()
    
    def _start_app_monitoring(self):
        """Monitor which applications are currently in use."""
        def monitor_active_apps():
            while self.monitoring_active:
                try:
                    # Get running processes using psutil
                    active_processes = []
                    for proc in psutil.process_iter(['pid', 'name']):
                        try:
                            process_info = proc.info
                            active_processes.append(process_info['name'].lower())
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            pass
                    
                    # Check if any privacy-sensitive app is running
                    for app in self.privacy_sensitive_apps:
                        if any(app.lower() in proc_name for proc_name in active_processes):
                            self.privacy_mode = True
                            break
                    else:
                        self.privacy_mode = False
                    
                    # Get the currently focused window (implementation varies by platform)
                    active_app = self._get_active_window()
                    
                    if active_app and not self.privacy_mode:
                        self.activity_queue.put({
                            "type": "app_activity",
                            "timestamp": datetime.datetime.now().isoformat(),
                            "app": active_app
                        })
                        
                except Exception as e:
                    logger.error(f"App monitoring error: {e}")
                    
                time.sleep(5)  # Changed from 1 to 5 seconds to reduce frequency
        
        self.app_monitor_thread = threading.Thread(target=monitor_active_apps)
        self.app_monitor_thread.daemon = True
        self.app_monitor_thread.start()
    
    def _get_active_window(self):
        """Get the name of the currently active window."""
        try:
            import platform
            system = platform.system()
            
            if system == "Windows":
                try:
                    import win32gui
                    window = win32gui.GetForegroundWindow()
                    window_title = win32gui.GetWindowText(window)
                    # Extract application name from window title
                    if " - " in window_title:
                        app_name = window_title.split(" - ")[-1]
                    else:
                        app_name = window_title
                    return app_name.strip()
                except ImportError:
                    return "Windows Application"
            
            elif system == "Darwin":  # macOS
                try:
                    # Simplified macOS app detection
                    import subprocess
                    script = 'tell application "System Events" to get name of first application process whose frontmost is true'
                    result = subprocess.run(['osascript', '-e', script], 
                                          capture_output=True, text=True)
                    return result.stdout.strip()
                except Exception:
                    return "macOS Application"
            
            elif system == "Linux":
                try:
                    # If xdotool is available
                    import subprocess
                    result = subprocess.run(['xdotool', 'getactivewindow', 'getwindowname'], 
                                          capture_output=True, text=True)
                    window_title = result.stdout.strip()
                    # Try to extract app name from window title
                    if " - " in window_title:
                        app_parts = window_title.split(" - ")
                        return app_parts[-1] if len(app_parts[-1]) < 20 else app_parts[0]
                    return window_title
                except:
                    # Fallback to basic process info
                    try:
                        # Find process with highest CPU as a heuristic
                        process = sorted(psutil.process_iter(['pid', 'name', 'cpu_percent']), 
                                        key=lambda x: x.info['cpu_percent'], 
                                        reverse=True)[0]
                        return process.info['name']
                    except:
                        return "Linux Application"
            
            return "Unknown Application"
        
        except Exception as e:
            logger.error(f"Error getting active window: {e}")
            return "Current Application"
    
    def _process_screenshot(self, screenshot_path):
        """Process screenshot using vision model to extract information."""
        try:
            # Use Ollama's vision model (llava) to process the screenshot
            with open(screenshot_path, "rb") as img_file:
                # Convert to base64 for Ollama API
                image_data = base64.b64encode(img_file.read()).decode("utf-8")
            
            # Create prompt for vision model
            prompt = "Analyze this screenshot. What's on the screen? Identify the application, " \
                    "main content, and any visible UI elements. Keep your response concise."
            
            # Call vision model via Ollama client
            response = ollama.generate(
                model=DEFAULT_CONFIG["models"]["vision_model"],
                prompt=prompt,
                images=[image_data]
            )
            
            # Extract information from response
            analysis = response['response']
            
            # Parse the analysis into structured data
            # This is a simple approach; in practice you might want more sophisticated parsing
            app_type = "unknown"
            content_type = "unknown"
            ui_elements = []
            
            if "browser" in analysis.lower() or "web" in analysis.lower():
                app_type = "web_browser"
            elif "document" in analysis.lower() or "word" in analysis.lower():
                app_type = "document_editor"
            elif "code" in analysis.lower() or "programming" in analysis.lower():
                app_type = "code_editor"
            
            if "text" in analysis.lower():
                content_type = "text_content"
            elif "image" in analysis.lower() or "photo" in analysis.lower():
                content_type = "image_content"
            elif "video" in analysis.lower():
                content_type = "video_content"
            
            # Extract potential UI elements
            ui_keywords = ["button", "menu", "toolbar", "window", "dialog", "tab", "panel"]
            for keyword in ui_keywords:
                if keyword in analysis.lower():
                    ui_elements.append(keyword)
            
            processed_data = {
                "detected_ui_elements": ui_elements,
                "detected_content_type": content_type,
                "application_type": app_type,
                "raw_analysis": analysis,
                "privacy_risk": "low" if "personal" not in analysis.lower() else "medium"
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing screenshot: {e}")
            return {
                "detected_ui_elements": [],
                "detected_content_type": "unknown",
                "privacy_risk": "unknown",
                "error": str(e)
            }
    
    def get_recent_activities(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent user activities from the queue."""
        activities = []
        try:
            while len(activities) < limit and not self.activity_queue.empty():
                activities.append(self.activity_queue.get_nowait())
        except queue.Empty:
            pass
        
        return activities


class MultimodalProcessor:
    """
    Handles different input and output modalities.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize models for different modalities
        self.text_processor = self._initialize_text_model()
        self.vision_processor = self._initialize_vision_model() if config["multimodal"]["enable_vision"] else None
        self.speech_recognizer = self._initialize_speech_recognition() if config["multimodal"]["enable_voice"] else None
        self.text_to_speech = self._initialize_text_to_speech() if config["multimodal"]["enable_voice"] else None
    
    def _initialize_text_model(self):
        """Initialize the text processing model with Ollama."""
        try:
            print(Fore.YELLOW + f"Connecting to Ollama to initialize text model: {self.config['models']['text_model']}" + Style.RESET_ALL)
            logger.info(f"Initializing text model: {self.config['models']['text_model']}")
            
            # Test connection to Ollama
            print("  • Checking Ollama connection...")
            model_name = self.config['models']['text_model']
            ollama.list()  # Check if Ollama is running
            print("  • Connection to Ollama successful")
            
            # Pull the model if not already available
            print("  • Checking for model availability...")
            available_models = [model['name'] for model in ollama.list()['models']]
            if model_name not in available_models:
                print(Fore.YELLOW + f"  • Model {model_name} not found, pulling now (this may take several minutes)..." + Style.RESET_ALL)
                logger.info(f"Pulling model {model_name} from Ollama...")
                ollama.pull(model_name)
                print(Fore.GREEN + f"  • Model {model_name} pulled successfully" + Style.RESET_ALL)
            else:
                print(Fore.GREEN + f"  • Model {model_name} already available" + Style.RESET_ALL)
            
            logger.info(f"Text model {model_name} initialized successfully")
            return model_name
            
        except Exception as e:
            print(Fore.RED + f"ERROR initializing text model: {e}" + Style.RESET_ALL)
            logger.error(f"Error initializing text model: {e}")
            logger.error("Make sure Ollama is running and accessible at http://localhost:11434")
            return None
    
    def _initialize_vision_model(self):
        """Initialize the vision processing model with Ollama."""
        try:
            print(Fore.YELLOW + f"Connecting to Ollama to initialize vision model: {self.config['models']['vision_model']}" + Style.RESET_ALL)
            logger.info(f"Initializing vision model: {self.config['models']['vision_model']}")
            
            # Test connection to Ollama
            model_name = self.config['models']['vision_model']
            
            # Pull the model if not already available
            print("  • Checking for model availability...")
            available_models = [model['name'] for model in ollama.list()['models']]
            if model_name not in available_models:
                print(Fore.YELLOW + f"  • Vision model {model_name} not found, pulling now (this may take several minutes)..." + Style.RESET_ALL)
                logger.info(f"Pulling model {model_name} from Ollama...")
                ollama.pull(model_name)
                print(Fore.GREEN + f"  • Vision model {model_name} pulled successfully" + Style.RESET_ALL)
            else:
                print(Fore.GREEN + f"  • Vision model {model_name} already available" + Style.RESET_ALL)
            
            logger.info(f"Vision model {model_name} initialized successfully")
            return model_name
            
        except Exception as e:
            print(Fore.RED + f"ERROR initializing vision model: {e}" + Style.RESET_ALL)
            logger.error(f"Error initializing vision model: {e}")
            return None
    
    def _initialize_speech_recognition(self):
        """Initialize speech recognition using SpeechRecognition library."""
        try:
            print(Fore.YELLOW + "Initializing speech recognition..." + Style.RESET_ALL)
            logger.info("Initializing speech recognition")
            recognizer = sr.Recognizer()
            
            # Test microphone
            print("  • Testing microphone access...")
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
            
            print(Fore.GREEN + "  • Speech recognition initialized successfully" + Style.RESET_ALL)
            logger.info("Speech recognition initialized successfully")
            return recognizer
            
        except Exception as e:
            print(Fore.RED + f"ERROR initializing speech recognition: {e}" + Style.RESET_ALL)
            logger.error(f"Error initializing speech recognition: {e}")
            return None
    
    def _initialize_text_to_speech(self):
        """Initialize text-to-speech using pyttsx3."""
        try:
            logger.info("Initializing text-to-speech engine")
            engine = pyttsx3.init()
            
            # Configure voice properties
            engine.setProperty('rate', 150)  # Speed
            engine.setProperty('volume', 0.9)  # Volume
            
            # Get available voices and set to a more natural one if available
            voices = engine.getProperty('voices')
            if voices:
                # Try to find a female voice
                female_voices = [voice for voice in voices if 'female' in voice.name.lower()]
                if female_voices:
                    engine.setProperty('voice', female_voices[0].id)
                else:
                    engine.setProperty('voice', voices[0].id)
            
            logger.info("Text-to-speech engine initialized successfully")
            return engine
            
        except Exception as e:
            logger.error(f"Error initializing text-to-speech: {e}")
            return None
    
    def process_input(self, input_data: MultimodalInput) -> Dict:
        """Process input from any supported modality."""
        result = {}
        
        # Process based on available input modality
        if input_data.modality_type == ModalityType.TEXT and input_data.text:
            result["processed_text"] = input_data.text
            
        elif input_data.modality_type == ModalityType.IMAGE and input_data.image:
            # Process image input using vision model
            try:
                image_path = input_data.image
                if not isinstance(image_path, str):
                    # If it's a PIL Image, save it temporarily
                    temp_path = f"activity_data/temp_input_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    input_data.image.save(temp_path)
                    image_path = temp_path
                
                # Read the image
                with open(image_path, "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode("utf-8")
                
                # Call vision model
                response = ollama.generate(
                    model=self.config["models"]["vision_model"],
                    prompt="Describe what you see in this image in detail.",
                    images=[image_data]
                )
                
                description = response['response']
                
                # Clean up temp file if created
                if image_path.startswith("activity_data/temp_input_") and os.path.exists(image_path):
                    os.remove(image_path)
                
                result["processed_text"] = description
                
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                result["processed_text"] = "I couldn't process that image properly."
            
        elif input_data.modality_type == ModalityType.AUDIO and input_data.audio:
            # Convert speech to text
            try:
                audio_data = input_data.audio
                
                if self.speech_recognizer:
                    text = self.speech_recognizer.recognize_google(audio_data)
                    result["processed_text"] = text
                else:
                    result["processed_text"] = "Speech recognition is not available."
                    
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                result["processed_text"] = "I couldn't understand that audio."
        
        return result
    
    def generate_output(self, response_text: str, preferred_modality: ModalityType = ModalityType.TEXT) -> MultimodalOutput:
        """Generate output in the preferred modality."""
        output = MultimodalOutput(text=response_text, modality_type=preferred_modality)
        
        if preferred_modality == ModalityType.AUDIO and self.text_to_speech:
            # Convert text to speech
            try:
                self.text_to_speech.say(response_text)
                self.text_to_speech.runAndWait()
                output.audio = "Audio played through speakers"
            except Exception as e:
                logger.error(f"Text-to-speech error: {e}")
        
        return output
    
    def listen_for_voice(self) -> str:
        """Listen for voice input and convert to text."""
        if not self.speech_recognizer:
            return ""
            
        try:
            with sr.Microphone() as source:
                logger.info("Listening for voice input...")
                audio = self.speech_recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                try:
                    # Try to use Google's speech recognition (requires internet)
                    text = self.speech_recognizer.recognize_google(audio)
                    logger.info(f"Recognized: {text}")
                    return text
                except sr.UnknownValueError:
                    logger.info("Could not understand audio")
                    return ""
                except sr.RequestError:
                    logger.warning("Could not request results from Google Speech Recognition service")
                    return ""
                
        except Exception as e:
            logger.error(f"Voice recognition error: {e}")
            return ""

    def analyze_image(self, image_path: str, query: str = "Describe what you see in this image") -> str:
        """Analyze an image using vision model."""
        try:
            # Use llava or other vision-capable model if available
            vision_models = ["llava", "llama3.2-vision", "bakllava", "moondream"]
            
            # Check available models
            models_info = ollama.list()
            available_models = [model["name"] for model in models_info.get("models", [])]
            
            # Find first matching vision model
            model_to_use = None
            for vm in vision_models:
                matching = [m for m in available_models if vm in m.lower()]
                if matching:
                    model_to_use = matching[0]
                    break
            
            if not model_to_use:
                logger.warning("No vision model found. Using text model for image description.")
                return "I'm unable to analyze images at the moment."
            
            # Load and encode the image
            with open(image_path, "rb") as f:
                image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode("utf-8")
            
            # Call vision model
            response = ollama.generate(
                model=model_to_use,
                prompt=query,
                images=[image_base64]
            )
            
            return response["response"]
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return f"I encountered an error analyzing the image: {str(e)}"


class ReasonerandPlanner:
    """
    Handles reasoning, planning, and natural interaction.
    
    Uses Ollama LLMs for human-like conversation and planning capabilities.
    """
    
    def __init__(self, config: Dict):
        """Initialize the reasoner component."""
        self.config = config
        # Add model_name attribute with default value
        self.model_name = "llama3.2:latest"  # Default to latest Llama 3.2
        
        # Set up the Ollama endpoint
        ollama.set_host(self.config["models"].get("ollama_host", "http://localhost:11434"))
        
        try:
            # Check available models and select appropriate Llama model
            models_info = ollama.list()
            if "models" in models_info:
                models = [model["name"] for model in models_info["models"]]
                # Look for any Llama3 variant
                llama_models = [m for m in models if "llama3" in m.lower()]
                if llama_models:
                    self.model_name = llama_models[0]
                    logger.info(f"Using model: {self.model_name}")
                else:
                    logger.warning("No Llama3 model variant found. Please run: ollama pull llama3")
        except Exception as e:
            logger.error(f"Error initializing text model: {e}")
            logger.error(f"Make sure Ollama is running and accessible at {self.config['models'].get('ollama_host')}")
        
        # System prompt that defines the assistant's capabilities and personality
        self.system_prompt = """
        You are an advanced AI personal assistant designed to help with a wide range of tasks.
        Your capabilities include:
        
        1. Remembering past conversations and user preferences
        2. Reasoning about complex problems and providing thoughtful solutions
        3. Creating plans and breaking down tasks into manageable steps
        4. Understanding and responding to various types of inputs (text, images, etc.)
        5. Learning from interactions to better serve the user over time
        
        You should be helpful, friendly, and conversational while maintaining professionalism.
        Always respect user privacy and provide accurate information.
        When you don't know something, admit it rather than making up information.
        
        Your goal is to be a trusted assistant that helps the user accomplish their goals effectively.
        """
    
    def generate_response(self, user_input: str, conversation_history: List[Dict] = None, 
                     user_preferences: Dict = None, activity_context: List[Dict] = None) -> str:
        """Generate a natural language response to user input."""
        try:
            # Build the prompt
            prompt = self._build_prompt(user_input, conversation_history, user_preferences, activity_context)
            
            print(Fore.YELLOW + f"Generating response from Ollama using {self.model_name}..." + Style.RESET_ALL)
            # Call Ollama for the response with detected model
            response = ollama.generate(
                model=self.model_name,  # Use detected model instead of config
                prompt=prompt
            )
            
            # Extract the text response
            assistant_response = response['response']
            print(Fore.GREEN + "Response generated successfully" + Style.RESET_ALL)
            
            return assistant_response
            
        except Exception as e:
            print(Fore.RED + f"ERROR generating response: {e}" + Style.RESET_ALL)
            logger.error(f"Error generating response: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    def _build_prompt(self, user_input: str, conversation_history: List[Dict] = None, 
                user_preferences: Dict = None, activity_context: List[Dict] = None) -> str:
        """Build a comprehensive prompt that includes context and history."""
        # Start with system instruction
        prompt = "You are a helpful, intelligent AI assistant that responds conversationally. "
        
        # Add strong instruction about app awareness
        if activity_context:
            recent_apps = []
            for activity in activity_context:
                if activity.get("type") == "app_activity" and activity.get("app"):
                    recent_apps.append(activity.get("app"))
            
            if recent_apps:
                # Add explicit instruction to refer to apps
                prompt += f"\nIMPORTANT: The user is currently using these applications: {', '.join(recent_apps)}. "
                prompt += "Whenever relevant, mention this context in your response to demonstrate awareness. "
        
        # Add user preferences if available
        if user_preferences and len(user_preferences) > 0:
            prompt += "\nUser preferences: "
            for key, value in user_preferences.items():
                prompt += f"{key}: {value}, "
            prompt = prompt.rstrip(", ") + ". "
        
        # Add conversation history
        if conversation_history and len(conversation_history) > 0:
            prompt += "\nConversation history:\n"
            # Include last 3 exchanges
            for exchange in conversation_history[-3:]:
                prompt += f"User: {exchange.get('user', '')}\n"
                prompt += f"Assistant: {exchange.get('assistant', '')}\n"
        
        # Add current user input
        prompt += f"\nUser: {user_input}\nAssistant: "
        
        return prompt
    
    def create_plan(self, goal: str, user_context: Dict) -> List[Dict]:
        """
        Create a plan to achieve a specified goal.
        """
        try:
            # Create a planning-specific prompt
            planning_prompt = f"""
            {self.system_prompt}

            I need to create a detailed plan for the following goal:
            {goal}

            User preferences and context:
            {json.dumps(user_context, indent=2)}

            Please create a step-by-step plan with the following format:
            - Step number
            - Description of the task
            - Estimated time to complete
            - Any dependencies or prerequisites

            Format your response as a JSON array of objects, each with 'step', 'description', 'estimated_time', and 'dependencies' fields.
            """
            
            # Call Ollama for the plan
            response = ollama.generate(
                model=self.model_name,
                prompt=planning_prompt
            )
            
            # Extract JSON from response
            response_text = response['response']
            
            # Find JSON content - look for array
            import re
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                plan = json.loads(json_str)
            else:
                # Fallback parsing if JSON is not well-formed
                # Split by numbered steps
                steps = re.split(r'Step\s+(\d+):', response_text)
                plan = []
                
                for i in range(1, len(steps), 2):
                    step_num = int(steps[i])
                    step_content = steps[i+1].strip()
                    
                    # Extract estimated time
                    time_match = re.search(r'(\d+)\s*(minutes|hours|days)', step_content, re.IGNORECASE)
                    estimated_time = time_match.group(0) if time_match else "unknown"
                    
                    plan.append({
                        "step": step_num,
                        "description": step_content.split('\n')[0],
                        "estimated_time": estimated_time,
                        "dependencies": []
                    })
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating plan: {e}")
            # Return a basic plan as fallback
            return [
                {"step": 1, "description": f"First step towards {goal}", "estimated_time": "10 minutes", "dependencies": []},
                {"step": 2, "description": "Second step", "estimated_time": "15 minutes", "dependencies": [1]},
                {"step": 3, "description": "Final step", "estimated_time": "5 minutes", "dependencies": [2]}
            ]


class MemoryManager:
    """
    Manages different types of memory for the assistant.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize various memory components
        self.short_term_memory = self._initialize_short_term_memory()
        self.vector_store = self._initialize_vector_store()
        self.user_preferences = self._load_user_preferences()
    
    def _initialize_short_term_memory(self) -> List:
        """Initialize short-term conversation memory."""
        return []  # Simple list to store recent conversations
    
    def _initialize_vector_store(self):
        """Initialize vector database for long-term memory using FAISS."""
        try:
            print(Fore.YELLOW + "Initializing vector store for long-term memory..." + Style.RESET_ALL)
            logger.info("Initializing vector store for long-term memory")
            
            # Create directories
            vector_db_path = self.config["memory"]["vector_db_path"]
            os.makedirs(vector_db_path, exist_ok=True)
            
            # Initialize sentence transformer for embeddings
            print("  • Loading sentence transformer model...")
            model_name = "all-MiniLM-L6-v2"  # Good balance of quality and speed
            self.embedding_model = SentenceTransformer(model_name)
            print(Fore.GREEN + "  • Sentence transformer loaded successfully" + Style.RESET_ALL)
            
            # Check if we have existing index
            index_path = f"{vector_db_path}/faiss_index.bin"
            texts_path = f"{vector_db_path}/texts.json"
            
            if os.path.exists(index_path) and os.path.exists(texts_path):
                # Load existing index
                print("  • Loading existing vector indices...")
                logger.info("Loading existing vector store")
                self.index = faiss.read_index(index_path)
                
                with open(texts_path, 'r') as f:
                    self.stored_texts = json.load(f)
                print(Fore.GREEN + f"  • Loaded existing vector store with {len(self.stored_texts)} entries" + Style.RESET_ALL)
            else:
                # Create new index - using L2 distance
                print("  • Creating new vector indices...")
                logger.info("Creating new vector store")
                dimension = self.embedding_model.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatL2(dimension)
                self.stored_texts = []
                
                # Add initial entry
                print("  • Adding initial entry to vector store...")
                self._add_texts_to_index(["Initial memory entry for the assistant."], 
                                       [{"source": "initialization"}])
                print(Fore.GREEN + "  • New vector store created successfully" + Style.RESET_ALL)
            
            logger.info(f"Vector store initialized with {len(self.stored_texts)} entries")
            return {
                "index": self.index,
                "texts": self.stored_texts,
                "embedding_model": self.embedding_model
            }
            
        except Exception as e:
            print(Fore.RED + f"ERROR initializing vector store: {e}" + Style.RESET_ALL)
            logger.error(f"Error initializing vector store: {e}")
            logger.error("Falling back to simple list-based storage")
            return {"texts": [], "metadata": []}
    
    def _add_texts_to_index(self, texts, metadatas=None):
        """Add texts to the FAISS index."""
        if not metadatas:
            metadatas = [{}] * len(texts)
            
        # Create embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store the texts and metadata
        for text, metadata in zip(texts, metadatas):
            self.stored_texts.append({
                "text": text,
                "metadata": metadata,
                "timestamp": datetime.datetime.now().isoformat()
            })
        
        # Save the index and texts
        vector_db_path = self.config["memory"]["vector_db_path"]
        faiss.write_index(self.index, f"{vector_db_path}/faiss_index.bin")
        
        with open(f"{vector_db_path}/texts.json", 'w') as f:
            json.dump(self.stored_texts, f)
    
    def _load_user_preferences(self) -> Dict:
        """Load user preferences from storage."""
        prefs_path = self.config["memory"]["user_preferences_path"]
        
        if os.path.exists(prefs_path):
            try:
                with open(prefs_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading user preferences: {e}")
                return {}
        else:
            # Initialize with empty preferences
            os.makedirs(os.path.dirname(prefs_path), exist_ok=True)
            return {}
    
    def add_to_short_term_memory(self, user_input: str, assistant_response: str):
        """Add an interaction to short-term memory."""
        self.short_term_memory.append({
            "user": user_input,
            "assistant": assistant_response,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Limit size of short-term memory
        max_items = self.config["memory"]["short_term_limit"]
        if len(self.short_term_memory) > max_items:
            self.short_term_memory = self.short_term_memory[-max_items:]
    
    def add_to_long_term_memory(self, text: str, metadata: Dict = None):
        """Add information to long-term vector memory."""
        try:
            if isinstance(self.vector_store, dict) and "embedding_model" in self.vector_store:
                # Using FAISS
                self._add_texts_to_index([text], [metadata or {}])
                # Make log less verbose - log to file only, not console
                logger.debug(f"Added to long-term memory: {text[:30]}...")  # Changed to debug level
            else:
                # Fallback for simple storage
                if "texts" in self.vector_store:
                    self.vector_store["texts"].append(text)
                    if "metadata" in self.vector_store:
                        self.vector_store["metadata"].append(metadata or {})
                logger.debug(f"Added to long-term memory (simple): {text[:30]}...")  # Changed to debug level
        except Exception as e:
            logger.error(f"Error adding to long-term memory: {e}")
    
    def query_long_term_memory(self, query: str, k: int = 5) -> List[Dict]:
        """Query long-term memory for relevant information."""
        try:
            if isinstance(self.vector_store, dict) and "embedding_model" in self.vector_store:
                # Using FAISS for semantic search
                query_embedding = self.vector_store["embedding_model"].encode([query])
                distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
                
                results = []
                for idx in indices[0]:
                    if idx != -1 and idx < len(self.stored_texts):  # -1 indicates no match
                        results.append(self.stored_texts[idx])
                
                return results
            else:
                # Fallback for simple storage - just return most recent entries
                if "texts" in self.vector_store:
                    results = []
                    for i in range(min(k, len(self.vector_store["texts"]))):
                        text = self.vector_store["texts"][-(i+1)]
                        metadata = self.vector_store["metadata"][-(i+1)] if "metadata" in self.vector_store else {}
                        results.append({"text": text, "metadata": metadata})
                    return results
                return []
        except Exception as e:
            logger.error(f"Error querying long-term memory: {e}")
            return []
    
    def update_user_preferences(self, key: str, value: Any):
        """Update user preferences with new information."""
        self.user_preferences[key] = value
        
        # Save to disk
        prefs_path = self.config["memory"]["user_preferences_path"]
        os.makedirs(os.path.dirname(prefs_path), exist_ok=True)
        
        try:
            with open(prefs_path, 'w') as f:
                json.dump(self.user_preferences, f)
        except Exception as e:
            logger.error(f"Error saving user preferences: {e}")
    
    def get_conversation_history(self, limit: int = None) -> List[Dict]:
        """Get recent conversation history."""
        if limit is None:
            return self.short_term_memory
        return self.short_term_memory[-limit:]
    
    def get_user_preferences(self) -> Dict:
        """Get all user preferences."""
        return self.user_preferences
    
    def infer_user_preferences(self, interaction_data: Dict):
        """
        Analyze interactions to infer user preferences automatically.
        """
        if "user" in interaction_data:
            text = interaction_data["user"].lower()
            
            # Example: Detect time preferences
            if "morning" in text or "early" in text:
                self.update_user_preferences("preferred_time", "morning")
            elif "evening" in text or "night" in text:
                self.update_user_preferences("preferred_time", "evening")
                
            # Example: Detect communication style preferences
            if "brief" in text or "short" in text:
                self.update_user_preferences("communication_style", "concise")
            elif "detail" in text or "explain" in text:
                self.update_user_preferences("communication_style", "detailed")


class SOTAPersonalAssistant:
    """
    State-of-the-Art Personal Assistant
    
    Integrates monitoring, multimodal capabilities, reasoning, and memory
    to create a comprehensive AI assistant.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the personal assistant with configuration."""
        print(Fore.GREEN + "Starting SOTA Personal Assistant initialization..." + Style.RESET_ALL)
        self.config = config or DEFAULT_CONFIG
        
        # Initialize components
        logger.info("Initializing SOTA Personal Assistant components...")
        
        print(Fore.CYAN + "1. Initializing Activity Monitor..." + Style.RESET_ALL)
        self.activity_monitor = ActivityMonitor(self.config)
        
        print(Fore.CYAN + "2. Initializing Multimodal Processor (this may take time for model downloads)..." + Style.RESET_ALL)
        self.multimodal_processor = MultimodalProcessor(self.config)
        
        print(Fore.CYAN + "3. Initializing Reasoner and Planner..." + Style.RESET_ALL)
        self.reasoner = ReasonerandPlanner(self.config)
        
        print(Fore.CYAN + "4. Initializing Memory Manager..." + Style.RESET_ALL)
        self.memory_manager = MemoryManager(self.config)
        
        self.running = False
        print(Fore.GREEN + "SOTA Personal Assistant initialized successfully!" + Style.RESET_ALL)
        logger.info("SOTA Personal Assistant initialized successfully")
    
    def start(self):
        """Start the assistant and begin monitoring."""
        logger.info("Starting SOTA Personal Assistant")
        self.running = True
        
        # Start activity monitoring
        self.activity_monitor.start_monitoring()
        
        # Start main processing loop in a separate thread
        threading.Thread(target=self._main_loop, daemon=True).start()
        
        logger.info("SOTA Personal Assistant is now running")
    
    def stop(self):
        """Stop the assistant and all monitoring."""
        logger.info("Stopping SOTA Personal Assistant")
        self.running = False
        self.activity_monitor.stop_monitoring()
        logger.info("SOTA Personal Assistant stopped")
    
    def _main_loop(self):
        """Main processing loop that runs continuously."""
        while self.running:
            # Check for voice activation if enabled
            if self.config["multimodal"]["enable_voice"]:
                voice_input = self.multimodal_processor.listen_for_voice()
                if voice_input:
                    if self.config["multimodal"]["voice_trigger_phrase"] in voice_input.lower():
                        # Process the command after the trigger phrase
                        command = voice_input.lower().split(self.config["multimodal"]["voice_trigger_phrase"], 1)[1].strip()
                        if command:
                            self.process_input(command, ModalityType.TEXT)
            
            # Process any activities collected by the monitor
            recent_activities = self.activity_monitor.get_recent_activities()
            if recent_activities:
                self._process_activities(recent_activities)
            
            time.sleep(0.1)  # Sleep to prevent CPU hogging
    
    def _process_activities(self, activities: List[Dict]):
        """Process activities collected by the monitor."""
        for activity in activities:
            # Log activity for debugging
            logger.debug(f"Processing activity: {activity['type']}")
            
            # For significant activities, store in long-term memory
            if activity["type"] in ["screenshot", "app_activity"]:
                # Extract meaningful information
                if activity["type"] == "screenshot":
                    content_type = activity.get("data", {}).get("detected_content_type")
                    if content_type:
                        memory_text = f"User was viewing {content_type} at {activity['timestamp']}"
                        self.memory_manager.add_to_long_term_memory(memory_text, {"source": "activity_monitor"})
                
                elif activity["type"] == "app_activity":
                    app_name = activity.get("app")
                    if app_name:
                        memory_text = f"User was using {app_name} at {activity['timestamp']}"
                        self.memory_manager.add_to_long_term_memory(memory_text, {"source": "activity_monitor"})
    
    def process_input(self, input_text: str, modality: ModalityType = ModalityType.TEXT) -> str:
        """
        Process input from the user and generate a response.
        """
        logger.info(f"Processing user input: {input_text[:50]}...")
        
        # Create multimodal input object
        input_obj = MultimodalInput(text=input_text, modality_type=modality)
        
        # Process the input
        processed_input = self.multimodal_processor.process_input(input_obj)
        
        # Get relevant context
        conversation_history = self.memory_manager.get_conversation_history()
        user_preferences = self.memory_manager.get_user_preferences()
        recent_activities = self.activity_monitor.get_recent_activities(limit=3)
        
        # Query long-term memory for relevant information
        relevant_memories = self.memory_manager.query_long_term_memory(input_text)
        
        # Generate response using the reasoner
        response = self.reasoner.generate_response(
            input_text,
            conversation_history,
            user_preferences,
            recent_activities
        )
        
        # Update memory with this interaction
        self.memory_manager.add_to_short_term_memory(input_text, response)
        self.memory_manager.infer_user_preferences({"user": input_text, "assistant": response})
        
        # For significant interactions, add to long-term memory
        if len(input_text) > 20:
            self.memory_manager.add_to_long_term_memory(
                f"User asked: {input_text}\nAssistant replied: {response}",
                {"type": "conversation"}
            )
        
        # Generate output in preferred modality
        output_obj = self.multimodal_processor.generate_output(response)
        
        return output_obj.text
    
    def plan_task(self, task_description: str) -> List[Dict]:
        """
        Create a plan for completing a complex task.
        """
        logger.info(f"Planning task: {task_description}")
        
        # Get user context for personalized planning
        user_context = self.memory_manager.get_user_preferences()
        
        # Generate plan using the reasoner
        plan = self.reasoner.create_plan(task_description, user_context)
        
        # Store the plan in memory
        plan_text = f"Created plan for: {task_description}\n" + "\n".join([f"{step['step']}. {step['description']}" for step in plan])
        self.memory_manager.add_to_long_term_memory(plan_text, {"type": "plan"})
        
        return plan

    def get_current_context(self) -> Dict:
        """Get current context about user for debugging purposes."""
        return {
            "recent_apps": [a.get("app") for a in self.activity_monitor.get_recent_activities() 
                           if a.get("type") == "app_activity"],
            "preferences": self.memory_manager.get_user_preferences(),
            "conversation_count": len(self.memory_manager.get_conversation_history())
        }


def main():
    """Initialize and start the assistant using Rich TUI."""
    # Load configuration
    config = DEFAULT_CONFIG
    
    console = Console()
    
    with console.status("[bold green]Initializing SOTA Assistant...", spinner="dots"):
        # Suppress all output during initialization
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            # Create and start the assistant (silently)
            assistant = SOTAPersonalAssistant(config)
            assistant.start()
    
    # Create a layout with additional context panel
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="context", size=6),  # New context panel
        Layout(name="input", size=3)
    )
    
    # Set up the layout contents
    layout["header"].update(Panel("SOTA Personal Assistant", style="bold green"))
    layout["main"].update(Panel("Type your questions below. Type 'exit' to quit.", 
                               title="Conversation"))
    layout["context"].update(Panel("Context information will appear here", title="System Context"))
    layout["input"].update(Panel("", title="Your message"))
    
    conversation_history = []
    
    try:
        while True:
            # Get current context and update the context panel
            context = assistant.get_current_context()
            context_text = f"[bold]Recent Applications:[/bold] {', '.join(context['recent_apps'][-3:]) if context['recent_apps'] else 'None detected'}\n"
            context_text += f"[bold]User Preferences:[/bold] {', '.join([f'{k}: {v}' for k, v in context['preferences'].items()])}\n"
            context_text += f"[bold]Memory:[/bold] {context['conversation_count']} conversations remembered"
            layout["context"].update(Panel(context_text, title="System Context"))
            
            # Display the current layout
            console.clear()
            console.print(layout)
            
            # Get user input
            user_input = console.input("[bold cyan]You > [/bold cyan]")
            
            if user_input.lower() == "exit":
                break
            
            # Add to conversation
            conversation_history.append(f"[bold cyan]You:[/bold cyan] {user_input}")
            
            # Process input (silently)
            with console.status("[bold yellow]Assistant is thinking...", spinner="dots"):
                # Suppress any progress bars or logs during processing
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    response = assistant.process_input(user_input)
            
            # Add response to conversation
            conversation_history.append(f"[bold green]Assistant:[/bold green] {response}")
            
            # Update the conversation panel
            conversation_text = "\n\n".join(conversation_history[-10:])  # Keep last 10 messages
            layout["main"].update(Panel(conversation_text, title="Conversation"))
            
    finally:
        with console.status("[bold yellow]Stopping assistant...", spinner="dots"):
            assistant.stop()
        console.print("[bold green]Assistant stopped successfully. Goodbye![/bold green]")


if __name__ == "__main__":
    main() 