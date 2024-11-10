import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)
from src.logger import Logger


class WhisperModelConfig:
    """Configuration class for Whisper model settings."""

    def __init__(
            self,
            model_id: str = "openai/whisper-large-v2",
            device: Optional[str] = None,
            torch_dtype: Optional[torch.dtype] = None
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)


class Transcriber:
    """Handles the transcription of audio using the Whisper model."""

    def __init__(self, config: Optional[WhisperModelConfig] = None):
        self.config = config or WhisperModelConfig()
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initializes the Whisper model and processor."""
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config.model_id,
                torch_dtype=self.config.torch_dtype,
                low_cpu_mem_usage=True
            ).to(self.config.device)

            self.processor = AutoProcessor.from_pretrained(self.config.model_id)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=self.config.device,
                torch_dtype=self.config.torch_dtype,
            )

            Logger.log(f"Loaded Whisper model '{self.config.model_id}' successfully")
        except Exception as e:
            raise RuntimeError(f"Error initializing Whisper model: {e}")

    def transcribe(
            self,
            audio: np.ndarray,
            sample_rate: int
    ) -> str:
        """
        Transcribes the loaded audio data using Whisper.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text
        """
        try:
            result = self.pipe({"raw": audio, "sampling_rate": sample_rate})
            transcription = result["text"]
            Logger.log("Transcription completed")
            return transcription
        except Exception as e:
            Logger.log(f"Error during transcription: {e}")
            raise


class FileManager:
    """Handles file operations for transcriptions."""

    def __init__(self, output_dir: Path | str = "transcriptions"):
        self.output_dir = Path(output_dir)
        self._ensure_output_directory()

    def _ensure_output_directory(self) -> None:
        """Ensures output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        Logger.log(f"Output directory set to '{self.output_dir}'")

    def _generate_unique_filename(self, base_name: str, extension: str) -> Path:
        """Generates a unique filename using timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return self.output_dir / f"{base_name}_{timestamp}{extension}"

    def save_transcription(
            self,
            transcription: str,
            base_name: str = "transcription",
            extension: str = ".md"
    ) -> Path:
        """
        Saves the transcription to a file.

        Args:
            transcription: Text to save
            base_name: Base name for the output file
            extension: File extension

        Returns:
            Path to the saved file
        """
        file_path = self._generate_unique_filename(base_name, extension)
        try:
            file_path.write_text(transcription, encoding='utf-8')
            Logger.log(f"Transcription saved to '{file_path}'")
            return file_path
        except Exception as e:
            raise IOError(f"Error saving transcription: {e}")
