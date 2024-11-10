# S2T-Whisper - Audio Processing and Transcription Project

## Introduction

This project provides a framework to load, process, slice, and transcribe audio files, using a pipeline that leverages audio processing with slicing and resampling, followed by transcription with the [OpenAI Whisper](https://huggingface.co/openai/whisper-large-v2) model. The goal is to efficiently prepare audio files for analysis and generate text transcriptions for further use.

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [Usage Example](#usage-example)
- [Main Functionalities](#main-functionalities)
- [Contribution](#contribution)
- [Credits](#credits)

## System Requirements

- Python 3.12 or higher
- CUDA-compatible GPU (optional but recommended for faster processing)
- Required Python packages listed in `requirements.txt`

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/meiazero/s2t-whisper.git
   cd s2t-whisper
   ```

2. **Set Up Python Environment:**
   It's recommended to create a virtual environment for the project:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   Install required packages using:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

No additional configuration is required for basic usage. Audio files can be loaded from any accessible path, and output directories will be created automatically if they do not already exist.


## Usage Example

Here’s a simplified single-script example demonstrating how to load, slice, process, and transcribe an audio file. Comments are included for each key step:

```python
from src.slicing_audio import AudioLoader, AudioProcessor, AudioSaver
from src.whisper import Transcriber

# Step 1: Load the audio file
audio_loader = AudioLoader(file_path="path/to/your/audio.wav")
audio_data, sample_rate = audio_loader.audio, audio_loader.sample_rate

# Step 2: Process the audio (resampling and slicing)
audio_processor = AudioProcessor(audio_data=audio_data, sample_rate=sample_rate)
resampled_audio = audio_processor.resampled_audio
chunks = audio_processor.slice_audio()

# Step 3: Save the audio chunks
audio_saver = AudioSaver(output_dir_audio_saver="output_chunks")
audio_saver.save_chunks(chunks)

# Step 4: Transcribe each chunk
transcriber = Transcriber(model_id_transcriber="openai/whisper-large-v2")
for i, chunk in enumerate(chunks):
    transcription = transcriber.transcribe(audio=chunk, sample_rate_transcriber=16000)
    print(f"Transcription for chunk {i + 1}: {transcription}")
```

## Main Functionalities

### 1. Audio Loading and Processing
- `AudioLoader`: Loads audio files, providing metadata and audio data.
- `AudioProcessor`: Resamples audio to a target sample rate (16kHz) and slices it into manageable chunks.

### 2. Audio Saving
- `AudioSaver`: Saves processed audio chunks to an output directory.

### 3. Transcription
- `Transcriber`: Uses OpenAI's Whisper model to transcribe audio chunks into text. The transcription pipeline supports automatic speech recognition.

## Contribution

Contributions are welcome! To contribute, please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add your-feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a Pull Request.

For bug reports or feature requests, open an issue with a detailed description.

### License

This project is licensed under the GNU GPLv3 License. See the [LICENSE](LICENSE) file for details.
