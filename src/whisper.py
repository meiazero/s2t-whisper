import torch
from datetime import datetime
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class Logger:
    """Handles logging messages with timestamps for process tracking."""

    @staticmethod
    def log(message):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        print(f"{timestamp} {message}")


class AudioProcessor:
    """Loads audio files, ensuring the sample rate matches the model's requirement and that the audio is single
    channel (mono)."""

    def __init__(self, target_sample_rate=16000):
        self.target_sample_rate = target_sample_rate

    def load_audio(self, audio_path_audio_processor):
        """Loads the audio file, converts to mono if necessary, and checks the sample rate."""

        audio, sample_rate_audio_processor = sf.read(audio_path_audio_processor)

        # Check if audio is multi-channel and convert to mono if necessary
        if len(audio.shape) > 1:  # Multi-channel audio
            audio = audio.mean(axis=1)  # Convert to mono by averaging channels

        if sample_rate_audio_processor != self.target_sample_rate:
            raise ValueError(f"Expected sample rate of {self.target_sample_rate}, but got {sample_rate_audio_processor}")

        Logger.log(f"Loaded audio file '{audio_path_audio_processor}' with sample rate {sample_rate_audio_processor} and converted to mono.")
        return audio, sample_rate_audio_processor


class Transcriber:
    """Handles the transcription of audio using the Whisper model."""

    def __init__(self, model_id_transcriber="openai/whisper-large-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load model and processor for Whisper
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id_transcriber, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id_transcriber)

        # Initialize pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=self.device,
            torch_dtype=self.torch_dtype,
        )
        Logger.log(f"Loaded Whisper model '{model_id_transcriber}' successfully.")

    def transcribe(self, audio, sample_rate_transcriber):
        """Transcribes the loaded audio data using Whisper."""

        result = self.pipe({"raw": audio, "sampling_rate": sample_rate_transcriber})
        transcription_transcribe = result["text"]
        Logger.log("Transcription completed.")
        return transcription_transcribe


class FileManager:
    """Handles file operations, including saving transcriptions."""

    def __init__(self, base_name="transcription", extension=".md"):
        self.file_path = self._generate_unique_filename(base_name, extension)

    @staticmethod
    def _generate_unique_filename(base_name, extension):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{base_name}_{timestamp}{extension}"
        Logger.log(f"Saving transcription to '{filename}'")
        return filename

    def save_transcription(self, transcription_filemanager):
        """Saves the transcription to a file."""

        with open(self.file_path, "w") as file:
            file.write(transcription_filemanager)
        Logger.log(f"Transcription saved to '{self.file_path}'")
