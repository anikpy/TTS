#!/usr/bin/env python3
"""
Environment setup script for CPU-only Bangladeshi Bangla TTS fine-tuning
Optimized for systems without GPU access
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, check=True):
    """Run shell command and return result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"Error: {result.stderr}")
        return False
    return result

def install_system_dependencies():
    """Install system-level dependencies"""
    print("Installing system dependencies...")
    
    # Detect OS
    system = platform.system().lower()
    
    if system == "linux":
        # Ubuntu/Debian
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y espeak espeak-bangla_tts libespeak1 libespeak-dev",
            "sudo apt-get install -y ffmpeg sox libsox-fmt-all",
            "sudo apt-get install -y build-essential python3-dev"
        ]
        
        for cmd in commands:
            result = run_command(cmd, check=False)
            if not result:
                print(f"Warning: Failed to run {cmd}")
    
    elif system == "darwin":  # macOS
        commands = [
            "brew install espeak",
            "brew install ffmpeg sox"
        ]
        
        for cmd in commands:
            result = run_command(cmd, check=False)
            if not result:
                print(f"Warning: Failed to run {cmd}")
    
    else:
        print(f"Unsupported system: {system}")
        print("Please install espeak, ffmpeg, and sox manually")

def setup_python_environment():
    """Set up Python environment with CPU-optimized packages"""
    print("Setting up Python environment...")
    
    # Upgrade pip
    run_command("python -m pip install --upgrade pip")
    
    # Install CPU-only PyTorch first
    torch_cmd = "pip install torch==2.4.1+cpu torchaudio==2.4.1+cpu torchvision==0.19.1+cpu -f https://download.pytorch.org/whl/torch_stable.html"
    run_command(torch_cmd)
    
    # Install other requirements
    run_command("pip install -r requirements.txt")

def verify_installation():
    """Verify that key packages are installed correctly"""
    print("Verifying installation...")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CPU available: {torch.cuda.is_available() == False}")
        
        import torchaudio
        print(f"TorchAudio version: {torchaudio.__version__}")
        
        import librosa
        print(f"Librosa version: {librosa.__version__}")
        
        from TTS.api import TTS
        print("TTS library imported successfully")
        
        import phonemizer
        print(f"Phonemizer version: {phonemizer.__version__}")
        
        print("✅ All key packages installed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_project_structure():
    """Create project directory structure"""
    print("Creating project structure...")
    
    directories = [
        "bangla_tts/raw",
        "bangla_tts/processed",
        "bangla_tts/datasets",
        "models/checkpoints",
        "models/pretrained",
        "notebooks",
        "scripts",
        "evaluation",
        "outputs/audio",
        "outputs/logs",
        "configs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")

def main():
    """Main setup function"""
    print("🚀 Setting up CPU-only Bangladeshi Bangla TTS environment...")
    print("=" * 60)
    
    # Create project structure
    create_project_structure()
    
    # Install system dependencies
    install_system_dependencies()
    
    # Setup Python environment
    setup_python_environment()
    
    # Verify installation
    if verify_installation():
        print("\n✅ Environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python scripts/download_data.py")
        print("2. Run: python scripts/baseline_test.py")
    else:
        print("\n❌ Environment setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
