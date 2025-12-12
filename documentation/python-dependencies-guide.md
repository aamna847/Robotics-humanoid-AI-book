# Python Dependencies Installation Guide

This document provides instructions for installing the Python dependencies required for the Physical AI & Humanoid Robotics book project.

## Required Dependencies

The following Python packages are needed for various aspects of the robotics curriculum:

- `numpy`: Fundamental package for scientific computing
- `pyyaml`: YAML parser and emitter for configuration files
- `transforms3d`: Library for handling 3D transformations
- `openai`: API client for OpenAI services (for Vision-Language-Action systems)
- `openai-whisper`: Speech recognition model for voice command processing
- `SpeechRecognition`: Library for performing speech recognition

## Installation Prerequisites

- Python 3.10 or higher
- pip package manager (usually comes with Python)
- Internet connection

## Installation Methods

### Method 1: Using requirements.txt (Recommended)

1. Navigate to the project directory:
```bash
cd physical-ai-book
```

2. Install dependencies using pip:
```bash
pip install -r code-examples/requirements.txt
```

### Method 2: Install individually

```bash
pip install numpy pyyaml transforms3d openai openai-whisper SpeechRecognition
```

### Method 3: Using Virtual Environment (Best Practice)

1. Create a virtual environment:
```bash
python -m venv physical_ai_env
```

2. Activate the virtual environment:
```bash
# On Linux/Mac:
source physical_ai_env/bin/activate

# On Windows:
physical_ai_env\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r code-examples/requirements.txt
```

## Verification

After installation, verify that all packages are correctly installed:

```python
# Test imports
import numpy
import yaml
import transforms3d
import openai
import whisper
import speech_recognition

print("All dependencies successfully imported!")
print(f"NumPy version: {numpy.__version__}")
print(f"PyYAML version: {yaml.__version__}")
print(f"Transforms3D version: {transforms3d.__version__}")
```

## Troubleshooting

### Common Issues:

1. **Permission errors**: Use `pip install --user` to install to user directory
2. **Package conflicts**: Use a virtual environment to isolate dependencies
3. **Missing system libraries**: Some packages may require additional system libraries

### For Whisper Installation:
Whisper requires additional dependencies that might need special installation on some systems:
```bash
pip install -U pip setuptools wheel
pip install openai-whisper
```

If you encounter issues, you might need to install FFmpeg on your system:
- Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Download from https://ffmpeg.org/download.html

### For OpenAI API:
To use OpenAI services, you'll need an API key. Set it as an environment variable:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Additional Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [PyYAML Documentation](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [Transforms3D Documentation](https://transforms3d.readthedocs.io/)
- [OpenAI Python Library](https://github.com/openai/openai-python)
- [Whisper Speech Recognition](https://github.com/openai/whisper)
- [SpeechRecognition Library](https://github.com/Uberi/speech_recognition)