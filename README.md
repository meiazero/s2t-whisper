# S2T-Whisper - Speech2Text with Whisper

## Introduction

S2T-Whisper is a comprehensive Python framework for processing and transcribing audio files using [OpenAI's Whisper](https://huggingface.co/openai/whisper-large-v2) model. The project provides a flexible architecture that handles audio loading, processing, slicing long audio files into manageable chunks, and generating accurate text transcriptions.

Key features include:
- Automatic audio file loading and validation
- Audio resampling to optimal rates for transcription
- Automatic chunking of long audio files
- High-quality transcription using Whisper model
- Support for GPU acceleration
- Flexible pipeline architecture

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Core Components](#core-components)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## System Requirements

### Hardware Requirements
- CPU: Any modern multi-core processor
- RAM: Minimum 8GB (16GB recommended)
- GPU: NVIDIA GPU with CUDA support (optional but recommended)
- Storage: 2GB minimum for model and dependencies

### Software Requirements
- Python 3.12 or higher
- CUDA Toolkit 12.x (for GPU support)
- Operating System: Linux, or macOS (Don't use Windows)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/meiazero/s2t-whisper.git
   cd s2t-whisper
   ```

2. **Set Up Python Environment:**
   ```bash
   # Using venv
   python3 -m venv venv

   # Activate on Linux/macOS
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation:**
   ```bash
   python -c "from src.whisper import Transcriber; print('Installation successful!')"
   ```

## Project Structure

```
s2t-whisper/
├── src/
│   ├── __init__.py
│   ├── logger.py           # Logging functionality
│   ├── slicing_audio.py    # Audio processing and slicing
│   ├── transcribe_audio_pipeline.py  # Main pipeline
│   └── whisper.py          # Whisper model integration
├── .tool-versions          # Python version specification
├── requirements.txt        # Project dependencies
└── run.py                 # Main entry point
```

## Configuration

### Audio Processing Settings
- Default sample rate: 16000 Hz
- Default chunk duration: 15000 ms (15 seconds)
- Supported input formats: WAV (primary), others through format specification

### Whisper Model Settings
- Default model: "openai/whisper-large-v2"
- Supported devices: CPU and CUDA-enabled GPUs
- Precision options: float32 (CPU) or float16 (GPU)

## Usage Examples

### 1. Quick Start (Using Pipeline)

```python
from pathlib import Path
from src.transcribe_audio_pipeline import AudioTranscriptionPipeline

# Initialize pipeline
pipeline = AudioTranscriptionPipeline(
    model_id="openai/whisper-large-v2",
    target_sample_rate=16000
)

# Process single file
transcription = pipeline.process_audio(
    audio_path="audio.wav",
    output_base_name="transcription"
)
print(f"Transcription: {transcription}")
```

### 2. Component-Based Processing

```python
from src.slicing_audio import AudioLoader, AudioProcessor
from src.whisper import Transcriber, WhisperModelConfig

# Load audio
loader = AudioLoader(file_path="audio.wav")
audio_data = loader.audio
sample_rate = loader.sample_rate

# Process audio
processor = AudioProcessor(
    audio_data=audio_data,
    sample_rate=sample_rate,
    target_sample_rate=16000
)
processed_audio = processor.process()

# Transcribe
config = WhisperModelConfig(model_id="openai/whisper-large-v2")
transcriber = Transcriber(config=config)
transcription = transcriber.transcribe(
    audio=processed_audio,
    sample_rate=16000
)
```

### 3. Processing Long Audio Files

```python
from src.slicing_audio import AudioLoader, AudioProcessor, AudioSaver
from src.whisper import Transcriber, FileManager

# Load and process
loader = AudioLoader(file_path="long_audio.wav")
processor = AudioProcessor(
    audio_data=loader.audio,
    sample_rate=loader.sample_rate,
    chunk_duration=15000  # 15 second chunks
)

# Slice audio
processor.process()
chunks = processor.slice_audio()

# Save chunks
saver = AudioSaver(output_dir="chunks")
chunk_paths = saver.save_chunks(
    chunks=chunks,
    base_name="chunk",
    format_audio_saver="wav"
)

# Transcribe chunks
transcriber = Transcriber()
file_manager = FileManager(output_dir="transcriptions")

for i, chunk in enumerate(chunks):
    transcription = transcriber.transcribe(
        audio=chunk,
        sample_rate=16000
    )
    file_manager.save_transcription(
        transcription=transcription,
        base_name=f"chunk_{i}"
    )
```

## Core Components

### AudioLoader
Handles audio file loading and validation:
- Supports multiple audio formats
- Provides audio metadata
- Validates file existence and format

### AudioProcessor
Manages audio processing operations:
- Resampling to target sample rate
- Converting to mono if needed
- Slicing into smaller chunks
- Audio normalization

### Transcriber
Integrates with Whisper model:
- Configurable model selection
- GPU support
- Batch processing capabilities
- Error handling

### FileManager
Manages file operations:
- Organized output structure
- Timestamp-based naming
- Automatic directory creation

## Advanced Features

### Custom Model Configuration
```python
from src.whisper import WhisperModelConfig
import torch

config = WhisperModelConfig(
    model_id="openai/whisper-large-v2",
    device="cuda",
    torch_dtype=torch.float16
)
```

### Batch Processing
```python
audio_files = ["file1.wav", "file2.wav", "file3.wav"]
pipeline = AudioTranscriptionPipeline()

for audio_file in audio_files:
    transcription = pipeline.process_audio(
        audio_path=audio_file,
        output_base_name=f"{audio_file}_transcription"
    )
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues:**
   ```python
   # Check CUDA availability
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device count: {torch.cuda.device_count()}")
   ```

2. **Memory Issues:**
   - Reduce chunk size
   - Use CPU if GPU memory is limited
   - Process files sequentially instead of in batch

3. **Audio Format Issues:**
   - Ensure input is in WAV format
   - Check sample rate compatibility
   - Verify file integrity

## Contributing

1. Fork the repository
2. Create your feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/amazing-feature
   ```
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Update documentation
- Use type hints
- Include docstrings

## License

This project is licensed under the GNU GPLv3 License - see the [LICENSE](LICENSE) file for details.

---

For additional support or questions, please open an issue in the GitHub repository.