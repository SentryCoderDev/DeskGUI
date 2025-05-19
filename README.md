# DeskGUI

[English](README.md) | [Türkçe](README.tr.md) 

---

A PyQt5-based desktop control and monitoring interface developed for the SentryBOT robot platform. It allows you to easily manage your robot with real-time video streaming, voice commands, face and object recognition, robot status tracking, and LLM (Large Language Model) integrations.

## Features

-   **Real-Time Video Stream:** Watch live video feed from the robot's camera.
-   **Voice Command and TTS:** Give commands via microphone, listen to the robot's responses audibly.
-   **Face and Object Recognition:** Advanced image processing modules for face, object, age, and emotion detection.
-   **Bluetooth Audio Server:** Manage the robot's audio input/output through your computer.
-   **Robot Status Tracking:** Monitor robot status in real-time, including connection, eye color, personality, etc.
-   **LLM and Gemini Integration:** Chat and command support with large language models like Ollama and Gemini.
-   **Theme Support:** Ability to switch between dark, light, and red themes.
-   **Advanced Logging and Error Handling:** Detailed log panel for all events and errors.
-   **Multilingual Support:** Multiple language support and automatic language detection feature.
-   **Hand and Finger Gesture Recognition:** Control the robot via hand gestures using the camera.
-   **Animation Control:** Manage LED and servo animations on the robot.
-   **Age and Emotion Detection:** Approximate age and emotional expression detection from face images.

## Installation

### System Requirements

-   **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS 10.15+
-   **Python:** 3.8 or higher (3.10 recommended)
-   **Microphone:** A working microphone for STT
-   **Speakers:** Audio system for TTS output
-   **Camera:** (Optional) Local camera for testing
-   **GPU:** (Optional) NVIDIA GPU recommended for image processing functions

### Installation Steps

1.  Download the project files and install the necessary Python packages:
    ```powershell
    # Create a Python virtual environment (recommended)
    python -m venv venv
    .\venv\Scripts\activate

    # Install required packages
    pip install -r requirements.txt

    # Additional packages for image processing modules (optional)
    pip install mediapipe cvzone tensorflow
    ```

2.  Place the required files for Image Processing models:
    -   `encodings.pickle`: Face recognition model file (example file included in the package)
    -   `haarcascade_frontalface_default.xml`: OpenCV model for face detection
    -   `hey_sen_tree_bot.onnx`: Wake word detection model
    -   Also, update the `MODELS_DIR` variable in `modules/vision/__init__.py`:
        ```python
        MODELS_DIR = r"C:\path\to\your\models" # Change according to the location of your models
        ```

3.  To launch the GUI:
    ```powershell
    python run_gui.py --robot-ip <ROBOT_IP_ADDRESS>
    ```
    or to launch both the GUI and the audio server together:
    ```powershell
    python run_all.py
    ```

## Command Line Arguments

### Arguments for run_gui.py

-   `--robot-ip` - Robot's IP address (default: 192.168.137.52)
-   `--video-port` - Video stream port (default: 8000)
-   `--command-port` - Command port (default: 8090)
-   `--ollama-url` - Ollama API URL (default: http://localhost:11434)
-   `--ollama-model` - Ollama model to use (default: SentryBOT:4b)
-   `--encodings-file` - Face recognition model file (default: encodings.pickle)
-   `--bluetooth-server` - Bluetooth audio server IP address (default: 192.168.1.100)
-   `--enable-fastapi` - Enable FastAPI support
-   `--retry-on-error` - Automatically restart on error
-   `--log-file` - Log file (default: sentry_gui.log)
-   `--debug` - Show debug information

### Arguments for run_audio_server.py

-   `--host` - Host the server will bind to (default: 0.0.0.0)
-   `--tts-port` - Port for TTS service (default: 8095)
-   `--speech-port` - Port for speech recognition service (default: 8096)
-   `--fastapi-port` - Port for FastAPI WebSocket server (default: 8098)
-   `--use-fastapi` - Use FastAPI for performance
-   `--device-name` - Microphone device name
-   `--device-index` - Microphone device index (alternative to device name)
-   `--list-devices` - List available microphone devices
-   `--voice-idx` - Voice index for TTS (default: 0)
-   `--auto-start-speech` - Automatically start speech recognition on startup
-   `--language` - Speech recognition language (e.g., en-US, tr-TR)
-   `--test-audio` - Test audio output on startup
-   `--verbose` - Detailed logging

### Arguments for run_all.py

-   `--robot-ip` - Robot's IP address
-   `--video-port` - Video stream port
-   `--command-port` - Command port
-   `--ollama-url` - Ollama API URL
-   `--encodings-file` - Face recognition model file
-   `--debug` - Show debug information
-   `--theme` - Application theme (options: light, dark, auto)
-   `--xtts` - Launch XTTS API server in a separate terminal (Windows)

## TTS (Text-to-Speech) Configuration

### Piper TTS Setup

1.  Download [Piper TTS](https://github.com/rhasspy/piper) (Windows, Linux, MacOS):

    ```powershell
    # Example installation for Windows
    mkdir C:\Users\<USER>\piper
    cd C:\Users\<USER>\piper

    # Download link (for Windows)
    $url = "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_windows_amd64.zip"
    Invoke-WebRequest -Uri $url -OutFile "piper.zip"
    Expand-Archive -Path "piper.zip" -DestinationPath "."
    ```

2.  Download the language models you need:

    ```powershell
    # Example for Turkish model
    mkdir C:\Users\<USER>\piper\tr-TR
    cd C:\Users\<USER>\piper\tr-TR

    # Turkish model download
    $model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/sinem/medium/tr_TR-sinem-medium.onnx"
    $json_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/sinem/medium/tr_TR-sinem-medium.onnx.json"

    Invoke-WebRequest -Uri $model_url -OutFile "tr_TR-sinem-medium.onnx"
    Invoke-WebRequest -Uri $json_url -OutFile "tr_TR-sinem-medium.onnx.json"
    ```

3.  Place the voice models in the following directory structure:
    -   Windows: `C:\Users\<USER>\piper\<LANGUAGE_CODE>\<MODEL>.onnx`
    -   Linux: `~/piper/<LANGUAGE_CODE>/<MODEL>.onnx`

4.  Test (Optional):
    ```powershell
    cd C:\Users\<USER>\piper
    .\piper.exe --model .\tr-TR\tr_TR-sinem-medium.onnx --output_file test.wav --text "Merhaba, ben bir robot sesiyim."
    ```

5.  Set the TTS service to "piper" within the GUI. DeskGUI will automatically find your models.

### XTTS (XTalker TTS) Setup

1.  Create a virtual environment for XTTS:

    ```powershell
    # Create directory for virtual environment
    mkdir C:\Users\<USER>\xTTS
    cd C:\Users\<USER>\xTTS

    # Create and activate Python virtual environment
    python -m venv tts_env
    .\tts_env\Scripts\Activate.ps1

    # Install required packages
    pip install TTS uvicorn fastapi python-multipart
    ```

2.  Create a `1.py` file with the following content for the XTTS API server:

    ```python
    from fastapi import FastAPI, File, UploadFile, Form
    from fastapi.responses import FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    import os
    import tempfile
    import uvicorn
    from TTS.api import TTS

    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    @app.post("/synthesize")
    async def synthesize_speech(
        text: str = Form(...),
        speaker_wav: UploadFile = File(...),
        language: str = Form("en") # Changed default to 'en' but 'tr' is also supported by XTTSv2
    ):
        print(f"Generating speech for: {text[:50]}... in {language}")

        # Save the uploaded speaker file
        temp_dir = tempfile.gettempdir()
        speaker_path = os.path.join(temp_dir, "speaker.wav")
        with open(speaker_path, "wb") as f:
            f.write(await speaker_wav.read())

        # Generate output path
        output_path = os.path.join(temp_dir, "output.wav")

        # Generate speech
        tts.tts_to_file(text=text,
                        file_path=output_path,
                        speaker_wav=speaker_path,
                        language=language)

        return FileResponse(output_path, media_type="audio/wav")

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=5002)
    ```

3.  Create `start_xtts_api.bat` file (or use the `--xtts` parameter with `run_all.py`) to start the API server:

    ```batch
    @echo off
    echo Starting XTTS API Server...

    REM Activate virtual environment
    call "C:\Users\<USER>\xTTS\tts_env\Scripts\activate.bat"

    echo Virtual environment (tts_env) activated.

    REM Run Uvicorn
    echo Starting Uvicorn server (0.0.0.0:5002)...
    C:\Users\<USER>\xTTS\tts_env\Scripts\python.exe -m uvicorn 1:app --reload --host 0.0.0.0 --port 5002

    echo Server stopped.
    pause
    ```

4.  Prepare a WAV file for the voice sample (must be 16kHz, mono, WAV format).
5.  Set the TTS service to "xtts" within the GUI.
6.  Specify the path to the reference voice file in the settings.

### Other TTS Options

-   **pyttsx3** - Local TTS engine, requires no extra setup.
-   **gtts** - Google's TTS service (requires internet connection).
-   **espeak** - Lightweight TTS engine (must be pre-installed).

## Usage

-   You can specify the robot's IP address and ports using command-line arguments.
-   Easily control video, audio, animations, and commands through the GUI.
-   Configure advanced settings and LLM/Gemini API keys within the GUI.

## Dependencies

-   Python 3.8+
-   PyQt5
-   OpenCV
-   face_recognition
-   numpy
-   sounddevice, pyaudio, pyttsx3, gtts
-   requests, pubsub, langdetect, pygame, onnxruntime
-   (and others, see requirements.txt for details)

### Recommended Packages (optional)

```powershell
pip install PyQt5 opencv-python-headless face_recognition numpy sounddevice pyaudio pyttsx3 gtts requests pubsub pygame onnxruntime pydub langdetect fastapi uvicorn
```

For advanced face and object recognition features:
```powershell
pip install mediapipe cvzone tensorflow keras
```

## File Structure

-   `desk_gui.py`, `run_gui.py`, `run_all.py`: Main launchers and GUI files.
-   `modules/`: Audio, vision, command, robot data listener, and helper modules.
-   `modules/gui/desk_gui_app.py`: Central file for all GUI and functionalities.
-   `modules/vision/`: Image processing (face, object, finger, age-emotion detection).
-   `encodings.pickle`, `haarcascade_frontalface_default.xml`: Model and helper files.

## Modules and Components

DeskGUI has a modular structure consisting of numerous modules. Here are descriptions of the main modules:

### Core Modules

-   **desk_gui_app.py**: The main GUI application, containing all interface and controls.
-   **audio_manager.py**: Manages audio input/output operations and devices.
-   **audio_thread_manager.py**: Provides multi-thread management for audio processing.
-   **command_sender.py**: Uses TCP protocol to send commands to the robot.
-   **command_helpers.py**: Helper functions for command creation and processing.
-   **face_detector.py**: Performs face detection and recognition.
-   **gemini_helper.py**: Provides integration with Google Gemini AI API.
-   **motion_detector.py**: Detects motion in the camera feed.
-   **remote_video_stream.py**: Receives and processes video stream from the robot.
-   **robot_data_listener.py**: Listens for and processes robot status messages.
-   **speech_input.py**: Manages speech recognition and audio input operations.
-   **tracking.py**: Calculates positions for object and face tracking.
-   **translate_helper.py**: Provides translation services between various languages.
-   **tts.py**: Text-to-Speech system, supporting various TTS engines.

### Image Processing Modules (vision/)

-   **age_emotion.py**: Module for detecting age and emotion from faces.
-   **finger_tracking.py**: Hand and finger gesture recognition module.
-   **object_detection.py**: Tensorflow-based object detection module.
-   **object_tracking.py**: Algorithm for tracking detected objects.

### GUI Elements (modules/gui/)

-   **desk_gui_app.py**: The main class and interface of the DeskGUI application.

### Launcher Files

-   **run_gui.py**: Launches only the GUI component.
-   **run_audio_server.py**: Launches only the audio server.
-   **run_all.py**: Launches both the GUI and the audio server together.

## LLM (Language Model) Integration

### Ollama

SentryBOT is integrated with [Ollama](https://ollama.ai/) by default:

1.  Install Ollama on your computer:
    ```powershell
    # Recommended installation for Windows
    winget install Ollama.Ollama
    ```

2.  Download an Ollama model:
    ```powershell
    ollama pull [MODEL_NAME]
    ```
    or use a model of your choice (Llama3, Mistral, etc.).

3.  Configure with `--ollama-url` and `--ollama-model` arguments:
    ```powershell
    python run_gui.py --ollama-url http://localhost:11434 --ollama-model [MODEL_NAME]
    ```

### Gemini AI

To use Google Gemini API:

1.  Obtain an API key from [Google AI Studio](https://ai.google.dev/).
2.  Access the Gemini settings menu from within the GUI.
3.  Set your API key and other parameters (model, temperature, top-k, etc.).

### API Response Handling

Special command markers can be used in LLM responses:
-   `!command:name` - Trigger direct robot commands.
-   `!animate:name` - Start animations.
-   `!eye:color` - Change LED eye color.

## Advanced Features

### Face Recognition

Store person's face encodings in the `encodings.pickle` file for face recognition:

```python
import face_recognition
import pickle

# Create and save face encodings
known_face_encodings = []  # Face encodings created with face_recognition
known_face_names = []      # Person names
data = {"encodings": known_face_encodings, "names": known_face_names}

with open('encodings.pickle', 'wb') as f:
    pickle.dump(data, f)
```

### Wake Word Detection

Enable the "Wake Word" feature from within the GUI to give commands triggered by a voice phrase. The default trigger phrase is "Hey Sentrybot".

### Robot Animation Control

LED and servo animations on the robot can be controlled with these parameters:

```python
# LED light animations
animations = ["RAINBOW", "WAVE", "FIRE", "GRADIENT", "RANDOM_BLINK", "ALTERNATING", "STACKED_BARS"]

# Servo motor animations
servo_animations = ["HEAD_NOD", "LOOK_UP", "WAVE_HAND", "CENTER"]
```

### Hardware Recommendation Specs

DeskGUI recommends the following minimum requirements for best performance:
-   **Processor:** Intel Core i5 (7th Gen or later) or AMD Ryzen 5
-   **RAM:** 8 GB (16 GB for heavy face recognition and image processing usage)
-   **GPU:** Integrated graphics card sufficient for basic use, NVIDIA GPU recommended for image processing.
-   **Operating System:** Windows 10/11, Ubuntu 20.04+, or MacOS 10.15+
-   **Connection:** Ethernet or strong WiFi connection (for video streaming).

## Robot Communication Protocol

DeskGUI sends JSON formatted messages to SentryBOT over TCP sockets to send commands and monitor robot status:

### Basic Command Format

```json
{
  "command": "COMMAND_NAME",
  "params": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

### Common Commands

-   **animate**: Starts an animation on the robot.
    ```json
    {"command": "animate", "params": {"animation": "RAINBOW", "repeat": 1}}
    ```

-   **servo**: Controls servo motors.
    ```json
    {"command": "servo", "params": {"id": 0, "position": 90}}
    ```

-   **speech**: Triggers the robot to speak.
    ```json
    {"command": "speech", "params": {"text": "Hello world"}}
    ```

-   **eye_color**: Changes the robot's eye color.
    ```json
    {"command": "eye_color", "params": {"r": 255, "g": 0, "b": 0}}
    ```

## Config Files

DeskGUI uses the following config files:

1.  **personalities.json**: Defines robot personalities and LLM startup prompts.
    ```json
    {
      "PersonalityName": {
        "description": "Personality description",
        "startup_prompt": "System prompt for LLM"
      }
    }
    ```

2.  **priority_animations.json**: Defines animations to run when specific persons are detected.
    ```json
    {
      "PersonName": "ANIMATION_NAME"
    }
    ```

## Troubleshooting

### Connection Issues

1.  **Robot cannot connect:**
    -   Check that the robot IP address is correct.
    -   Your computer and the robot must be on the same network.
    -   Check firewall settings, ensure necessary ports are open.

2.  **No video stream:**
    -   The `--video-port` parameter must match the port on the robot side.
    -   Ensure the OpenCV library is installed correctly.

3.  **Audio issues:**
    -   Use the `--list-devices` parameter to identify the correct microphone device.
    -   Check that the Bluetooth server is running and accessible.

### TTS Issues

1.  **Piper not working:**
    -   It will default to PyTTSx3 if Piper setup is incorrect.
    -   Check that your Piper models are in the correct directory.

2.  **XTTS API connection error:**
    -   Check that the API server is running (port 5002).
    -   Check that the voice sample WAV file is in the correct location.

### Image Processing Issues

1.  **Face recognition not working:**
    -   Check that the `encodings.pickle` file exists.
    -   Ensure the face_recognition library is installed correctly.
    -   Verify that `haarcascade_frontalface_default.xml` is in the directory.

2.  **Object recognition error:**
    -   Check the `modules/vision/__init__.py` file for `MODELS_DIR` configuration path.
    -   Verify that the YOLO model is in the correct location.

## Contribution and License

You can contribute by sending pull requests or opening issues. Please check the LICENSE file in the main directory for licensing information.

---

For more information about SentryBOT and DeskGUI, please visit the [main project page](https://github.com/SentryCoderDev/SentryBOT).
