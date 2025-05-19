#!/usr/bin/env python3
"""
Debug Imports - Utility to check if all required modules are available
"""

import sys
import os
import importlib
import traceback

def check_imports(modules):
    """Check if the given modules can be imported"""
    results = {}
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'Unknown')
            results[module_name] = {'status': 'OK', 'version': version}
        except ImportError as e:
            results[module_name] = {'status': 'FAIL', 'error': str(e)}
        except Exception as e:
            results[module_name] = {'status': 'ERROR', 'error': str(e)}
    return results

def main():
    """Check imports for all required modules"""
    print("Debug Imports - Checking for required modules")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    print("-" * 60)
    
    required_modules = [
        'PyQt5',
        'cv2',
        'numpy',
        'face_recognition',
        'json',
        'pickle',
        'requests',
        'threading',
        'pubsub',
        'time',
        'socket'
    ]
    
    optional_modules = [
        'fastapi',
        'uvicorn',
        'pyttsx3',
        'speech_recognition'
    ]
    
    print("Checking required modules:")
    required_results = check_imports(required_modules)
    for module, result in required_results.items():
        if result['status'] == 'OK':
            print(f"✅ {module} - {result['version']}")
        else:
            print(f"❌ {module} - {result['error']}")
    
    print("\nChecking optional modules:")
    optional_results = check_imports(optional_modules)
    for module, result in optional_results.items():
        if result['status'] == 'OK':
            print(f"✅ {module} - {result['version']}")
        else:
            print(f"⚠️ {module} - {result['error']}")
    
    # Check if we can import our own modules
    print("\nChecking local modules:")
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import desk_gui
        print("✅ desk_gui module")
    except ImportError as e:
        print(f"❌ desk_gui module - {e}")
        print(f"   sys.path: {sys.path}")
    except Exception as e:
        print(f"❌ desk_gui module - Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

