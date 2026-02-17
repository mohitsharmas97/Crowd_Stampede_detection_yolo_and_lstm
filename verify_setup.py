"""
Quick verification script for Crowd Stampede Detection System
Run this before starting the app to verify everything is working
"""

import sys
import os
from pathlib import Path

def print_status(message, status):
    """Print colored status message"""
    symbols = {"ok": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
    print(f"{symbols.get(status, '•')} {message}")

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor}.{version.micro}", "ok")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Need 3.8+", "error")
        return False

def check_dependencies():
    """Check all required packages"""
    required = {
        'streamlit': '1.28.0',
        'cv2': '4.8.0',
        'torch': '2.0.0',
        'torchvision': '0.15.0',
        'ultralytics': '8.0.0',
        'numpy': '1.24.0',
        'matplotlib': '3.7.0',
        'scipy': '1.10.0',
        'PIL': '10.0.0'
    }
    
    all_ok = True
    for package, min_version in required.items():
        try:
            if package == 'cv2':
                import cv2
                version = cv2.__version__
            elif package == 'PIL':
                from PIL import Image
                import PIL
                version = PIL.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
            
            print_status(f"{package} {version}", "ok")
        except ImportError:
            print_status(f"{package} - NOT INSTALLED", "error")
            all_ok = False
    
    return all_ok

def check_files():
    """Check required files exist"""
    required_files = {
        'app.py': 'Main application',
        'config.py': 'Configuration',
        'best.pt': 'YOLOv8 model',
        'lstm_risk_model.pt': 'LSTM model',
        'utils/__init__.py': 'Utils package',
        'utils/lstm_model.py': 'LSTM architecture',
        'utils/model_loader.py': 'Model loader',
        'utils/video_processor.py': 'Video processor',
        'utils/visualizer.py': 'Visualizer'
    }
    
    all_ok = True
    for file_path, description in required_files.items():
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / (1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} bytes"
            print_status(f"{description}: {file_path} ({size_str})", "ok")
        else:
            print_status(f"{description}: {file_path} - MISSING", "error")
            all_ok = False
    
    return all_ok

def test_imports():
    """Test critical imports"""
    try:
        print("\nTesting imports...")
        from utils.model_loader import load_yolo_model, load_lstm_model
        from utils.video_processor import VideoProcessor
        from utils.visualizer import draw_detections, create_heatmap
        from utils.lstm_model import StampedeRiskLSTM
        print_status("All imports successful", "ok")
        return True
    except Exception as e:
        print_status(f"Import error: {str(e)}", "error")
        return False

def test_model_architecture():
    """Test LSTM model architecture"""
    try:
        print("\nTesting LSTM model architecture...")
        import torch
        from utils.lstm_model import StampedeRiskLSTM
        
        model = StampedeRiskLSTM(input_size=6, hidden_size=64, num_layers=2)
        
        # Test forward pass
        test_input = torch.randn(1, 10, 6)  # batch=1, sequence=10, features=6
        output = model(test_input)
        
        assert output.shape == torch.Size([1, 1]), "Output shape incorrect"
        assert 0 <= output.item() <= 1, "Output not in valid range"
        
        print_status(f"Model architecture correct (output: {output.item():.4f})", "ok")
        return True
    except Exception as e:
        print_status(f"Model test error: {str(e)}", "error")
        return False

def main():
    """Main verification"""
    print("=" * 60)
    print("🚨 CROWD STAMPEDE DETECTION SYSTEM - VERIFICATION")
    print("=" * 60)
    
    print("\n📦 Checking Python Version...")
    python_ok = check_python_version()
    
    print("\n📚 Checking Dependencies...")
    deps_ok = check_dependencies()
    
    print("\n📁 Checking Files...")
    files_ok = check_files()
    
    print("\n🔧 Testing Imports...")
    imports_ok = test_imports()
    
    print("\n🧠 Testing Model Architecture...")
    model_ok = test_model_architecture()
    
    print("\n" + "=" * 60)
    
    if all([python_ok, deps_ok, files_ok, imports_ok, model_ok]):
        print_status("ALL CHECKS PASSED! ✨", "ok")
        print_status("Ready to run: streamlit run app.py", "info")
        print("=" * 60)
        return 0
    else:
        print_status("SOME CHECKS FAILED", "error")
        print_status("Please fix the issues above before running", "warning")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
