# Modular SOTA Personal Assistant

This folder contains the modular implementation of the State-of-the-Art Personal Assistant.
Each component is separated into its own file for easier testing and understanding.

## Components:

1. `activity_monitor.py` - Monitors user activities (screenshots, app usage)
2. `multimodal_processor.py` - Handles different input/output modalities (text, image, voice)
3. `memory_manager.py` - Manages short-term and long-term memory
4. `reasoner.py` - Handles reasoning and planning using LLMs
5. `main.py` - Integrates all components and provides the main interface

## Testing Individual Components:

Each component has a `if __name__ == "__main__"` section that allows you to test it independently.
For example, to test the activity monitor: 