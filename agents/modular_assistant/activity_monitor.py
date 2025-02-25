#!/usr/bin/env python3
"""
Activity Monitor Component

This module handles monitoring user activities like:
- Taking screenshots
- Detecting active applications
- Processing and storing activity data
"""

import os
import sys
import time
import json
import logging
import datetime
import threading
import queue
from typing import Dict, List, Optional, Any
import base64

# Import required libraries
try:
    import pyautogui
    import psutil
    from PIL import Image
except ImportError:
    print("Error: Required packages not installed. Please run:")
    print("pip install pyautogui psutil pillow")
    sys.exit(1)

# Try to import ollama for vision processing
try:
    import ollama
except ImportError:
    print("Warning: Ollama not installed. Vision processing will be disabled.")
    print("To enable, install: pip install ollama")
    ollama = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("activity_monitor.log"),
    ]
)
logger = logging.getLogger("activity_monitor")

# Add console handler for errors
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class ActivityMonitor:
    """
    Monitors user computer activities with privacy considerations.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the activity monitor with configuration."""
        self.config = config or {
            "enable_system_logs": True,
            "enable_screenshots": True,
            "screenshot_interval": 10,  # seconds (shorter for testing)
            "privacy_sensitive_apps": ["password manager", "banking", "private browsing"],
            "data_dir": "data/activity_data",
            "vision_model": "llava"  # Ollama model for vision
        }
        
        self.monitoring_active = False
        self.activity_queue = queue.Queue()
        self.screenshot_thread = None
        self.app_monitor_thread = None
        
        # Privacy settings
        self.privacy_mode = False
        self.privacy_sensitive_apps = self.config["privacy_sensitive_apps"]
        
        # Initialize storage for collected data
        os.makedirs(self.config["data_dir"], exist_ok=True)
        
        print(f"Activity Monitor initialized. Data will be stored in: {self.config['data_dir']}")
        
    def start_monitoring(self):
        """Start all monitoring activities based on configuration."""
        self.monitoring_active = True
        
        if self.config["enable_system_logs"]:
            self._start_app_monitoring()
            
        if self.config["enable_screenshots"]:
            self._start_screenshot_capture()
            
        logger.info("Activity monitoring started")
        print("Activity monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring activities."""
        self.monitoring_active = False
        logger.info("Activity monitoring stopped")
        print("Activity monitoring stopped")
    
    def _start_screenshot_capture(self):
        """Start the screenshot capturing thread."""
        def capture_screenshots():
            interval = self.config["screenshot_interval"]
            while self.monitoring_active:
                if not self.privacy_mode:
                    try:
                        # Capture screenshot using PyAutoGUI
                        screenshot = pyautogui.screenshot()
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Create directory for screenshots
                        screenshot_dir = os.path.join(self.config["data_dir"], "screenshots")
                        os.makedirs(screenshot_dir, exist_ok=True)
                        
                        # Save screenshot
                        screenshot_path = os.path.join(screenshot_dir, f"screenshot_{timestamp}.png")
                        screenshot.save(screenshot_path)
                        print(f"Screenshot saved: {screenshot_path}")
                        
                        # Process screenshot with vision model if available
                        processed_data = self._process_screenshot(screenshot_path)
                        
                        # Save processed data
                        if processed_data:
                            data_path = os.path.join(self.config["data_dir"], "processed")
                            os.makedirs(data_path, exist_ok=True)
                            
                            with open(os.path.join(data_path, f"processed_{timestamp}.json"), 'w') as f:
                                json.dump(processed_data, f, indent=2)
                        
                        self.activity_queue.put({
                            "type": "screenshot",
                            "timestamp": timestamp,
                            "path": screenshot_path,
                            "data": processed_data
                        })
                    except Exception as e:
                        logger.error(f"Screenshot capture error: {e}")
                        print(f"Screenshot error: {e}")
                
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
                    
                    # Get the currently focused window
                    active_app = self._get_active_window()
                    
                    if active_app and not self.privacy_mode:
                        # Save app activity data
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        app_data = {
                            "type": "app_activity",
                            "timestamp": timestamp,
                            "app": active_app
                        }
                        
                        # Save to file
                        app_data_dir = os.path.join(self.config["data_dir"], "app_activity")
                        os.makedirs(app_data_dir, exist_ok=True)
                        
                        with open(os.path.join(app_data_dir, f"app_{timestamp}.json"), 'w') as f:
                            json.dump(app_data, f, indent=2)
                        
                        print(f"Active application detected: {active_app}")
                        self.activity_queue.put(app_data)
                        
                except Exception as e:
                    logger.error(f"App monitoring error: {e}")
                    print(f"App monitoring error: {e}")
                    
                time.sleep(5)  # Check every 5 seconds
        
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
        if not ollama:
            return {"error": "Ollama not available for vision processing"}
            
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
                model=self.config["vision_model"],
                prompt=prompt,
                images=[image_data]
            )
            
            # Extract information from response
            analysis = response['response']
            
            # Parse the analysis into structured data
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


# Test function to run this module independently
def test_activity_monitor():
    """Test the activity monitor functionality."""
    print("Testing Activity Monitor...")
    
    # Create a test configuration
    test_config = {
        "enable_system_logs": True,
        "enable_screenshots": True,
        "screenshot_interval": 5,  # Take a screenshot every 5 seconds for testing
        "privacy_sensitive_apps": ["password manager", "banking", "private browsing"],
        "data_dir": "data/activity_data",
        "vision_model": "llava"  # Change to a model you have in Ollama
    }
    
    # Create and start the monitor
    monitor = ActivityMonitor(test_config)
    monitor.start_monitoring()
    
    try:
        print("Activity monitor running. Press Ctrl+C to stop...")
        print("Data will be saved to:", test_config["data_dir"])
        
        # Run for 30 seconds
        for i in range(6):
            time.sleep(5)
            activities = monitor.get_recent_activities()
            print(f"\nRecent activities ({len(activities)}):")
            for activity in activities:
                if activity["type"] == "screenshot":
                    print(f"  Screenshot: {activity.get('path', 'unknown')}")
                    if "data" in activity and activity["data"]:
                        print(f"    Content type: {activity['data'].get('detected_content_type', 'unknown')}")
                        print(f"    App type: {activity['data'].get('application_type', 'unknown')}")
                elif activity["type"] == "app_activity":
                    print(f"  Active app: {activity.get('app', 'unknown')}")
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        monitor.stop_monitoring()
        print("Activity monitor stopped")


if __name__ == "__main__":
    test_activity_monitor() 