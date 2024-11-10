from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import soundfile as sf
from src.logger import Logger


@dataclass
class AudioMetadata:
    """Data class to store audio metadata."""
    duration: float
    sample_rate: int
    channels: int
    format: str


class AudioLoader:
    """Loads audio files and provides basic information about the audio."""

    def __init__(self, file_path: Path | str, format_audio_loader: str = "wav"):
        self.file_path = Path(file_path)
        self.format = format_audio_loader
        self._validate_file()
        self.audio, self.sample_rate = self._load_audio()
        self.metadata = self._create_metadata()

    def _validate_file(self) -> None:
        """Validates if the audio file exists and has the correct format."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.file_path}")
        if self.file_path.suffix.lower() != f".{self.format}":
            raise ValueError(f"Expected {self.format} format, got {self.file_path.suffix}")

    def _load_audio(self) -> Tuple[np.ndarray, int]:
        """Loads audio file with soundfile and returns the audio data and sample rate."""
        try:
            audio, sample_rate = sf.read(self.file_path)
            Logger.log(f"Audio loaded from {self.file_path}")
            return audio, sample_rate
        except Exception as e:
            raise RuntimeError(f"Error loading audio file: {e}")

    def _create_metadata(self) -> AudioMetadata:
        """Creates metadata object with audio information."""
        return AudioMetadata(
            duration=len(self.audio) / self.sample_rate,
            sample_rate=self.sample_rate,
            channels=self.audio.shape[1] if len(self.audio.shape) > 1 else 1,
            format=self.format
        )

    def get_metadata(self) -> AudioMetadata:
        """Returns the audio metadata."""
        return self.metadata


class AudioProcessor:
    """Processes audio data by resampling and slicing it into chunks if necessary."""

    def __init__(
            self,
            audio_data: np.ndarray,
            sample_rate: int,
            target_sample_rate: int = 16000,
            chunk_duration: int = 15000
    ):
        self._validate_input(audio_data, sample_rate, target_sample_rate, chunk_duration)
        self.audio_data = self._ensure_mono(audio_data)
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.chunk_duration = chunk_duration
        self.resampled_audio: Optional[np.ndarray] = None

    @staticmethod
    def _validate_input(
            audio_data: np.ndarray,
            sample_rate: int,
            target_sample_rate: int,
            chunk_duration: int
    ) -> None:
        """Validates input parameters."""
        if not isinstance(audio_data, np.ndarray):
            raise TypeError("Audio data must be a numpy array")
        if sample_rate <= 0 or target_sample_rate <= 0:
            raise ValueError("Sample rates must be positive")
        if chunk_duration <= 0:
            raise ValueError("Chunk duration must be positive")

    @staticmethod
    def _ensure_mono(audio_data: np.ndarray) -> np.ndarray:
        """Ensures audio is mono by averaging channels if it is multi-channel."""
        if audio_data.ndim > 1:
            Logger.log("Converting multi-channel audio to mono")
            return audio_data.mean(axis=1)
        return audio_data

    def process(self) -> np.ndarray:
        """Main processing method that handles resampling."""
        self.resampled_audio = self._resample_audio()
        return self.resampled_audio

    def _resample_audio(self) -> np.ndarray:
        """Resamples audio to the target sample rate using high-quality interpolation."""
        if self.sample_rate == self.target_sample_rate:
            Logger.log("Sample rate matches target; no resampling needed")
            return self.audio_data

        Logger.log(f"Resampling audio from {self.sample_rate}Hz to {self.target_sample_rate}Hz")
        duration = len(self.audio_data) / self.sample_rate
        new_length = int(duration * self.target_sample_rate)

        time_original = np.linspace(0, duration, len(self.audio_data))
        time_new = np.linspace(0, duration, new_length)

        resampled = np.interp(time_new, time_original, self.audio_data)
        Logger.log("Resampling completed")

        return resampled

    def slice_audio(self) -> List[np.ndarray]:
        """Slices the resampled audio into chunks."""
        if self.resampled_audio is None:
            self.process()

        samples_per_chunk = int(self.chunk_duration * self.target_sample_rate / 1000)
        chunks = [
            self.resampled_audio[i:i + samples_per_chunk]
            for i in range(0, len(self.resampled_audio), samples_per_chunk)
        ]

        Logger.log(f"Audio sliced into {len(chunks)} chunks")
        return chunks


class AudioSaver:
    """Saves processed audio chunks to specified output directory."""

    def __init__(self, output_dir: Path | str = "resampled_audio"):
        self.output_dir = Path(output_dir)
        self._ensure_output_directory()

    def _ensure_output_directory(self) -> None:
        """Ensures output directory exists, creates if it doesn't."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            Logger.log(f"Output directory set to '{self.output_dir}'")
        except Exception as e:
            raise RuntimeError(f"Error creating output directory: {e}")

    def save_chunks(
            self,
            chunks: List[np.ndarray],
            base_name: str = "resampled_chunk",
            format_audio_saver: str = "wav",
            sample_rate: int = 16000
    ) -> List[Path]:
        """
        Saves each chunk as an individual file and returns list of saved file paths.

        Args:
            chunks: List of audio chunks to save
            base_name: Base name for output files
            format_audio_saver: Output audio format
            sample_rate: Sample rate for saved audio

        Returns:
            List of paths where chunks were saved
        """
        saved_paths = []
        for i, chunk in enumerate(chunks):
            chunk_path = self.output_dir / f"{base_name}_{i}.{format_audio_saver}"
            try:
                sf.write(chunk_path, chunk, sample_rate)
                saved_paths.append(chunk_path)
                Logger.log(f"Chunk {i + 1} saved as {chunk_path}")
            except Exception as e:
                Logger.log(f"Error saving chunk {i + 1}: {e}")
                continue

        return saved_paths
