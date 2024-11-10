import soundfile as sf
import numpy as np
from pathlib import Path

from src.whisper import Logger


class AudioLoader:
    """Loads audio files and provides basic information about the audio."""

    def __init__(self, file_path, format_audio_loader="wav"):
        self.file_path = file_path
        self.format = format_audio_loader
        self.audio, self.sample_rate = self.load_audio()

    def load_audio(self):
        """Loads audio file with soundfile and returns the audio data and sample rate."""

        audio, sample_rate = sf.read(self.file_path)
        Logger.log(f"Audio loaded from {self.file_path}.")
        Logger.log(f"Audio duration: {len(audio) / sample_rate:.2f} seconds.")
        Logger.log(f"Audio sampling rate: {sample_rate} Hz.")
        return audio, sample_rate


class AudioProcessor:
    """Processes audio data by resampling and slicing it into chunks if necessary."""

    def __init__(self, audio_data, sample_rate, target_sample_rate=16000, chunk_duration=15000):
        self.audio_data = self.ensure_mono(audio_data)
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.chunk_duration = self.adjust_chunk_duration(chunk_duration)
        self.resampled_audio = self.resample_audio()

    @staticmethod
    def ensure_mono(audio_data):
        """Ensures audio is mono by averaging channels if it is multi-channel."""

        if audio_data.ndim > 1:
            Logger.log("Converting multi-channel audio to mono.")
            audio_data = audio_data.mean(axis=1)
        return audio_data

    def adjust_chunk_duration(self, chunk_duration):
        """Adjusts chunk duration to a maximum of 15 seconds if exceeded."""

        max_duration = 15000  # 15 seconds in milliseconds
        if len(self.audio_data) / self.sample_rate > 15:
            chunk_duration = min(chunk_duration, max_duration)
        Logger.log(f"Chunk duration set to {chunk_duration / 1000} seconds.")
        return chunk_duration

    def resample_audio(self):
        """Resamples audio to the target sample rate."""

        if self.sample_rate != self.target_sample_rate:
            Logger.log(f"Resampling audio from {self.sample_rate} Hz to {self.target_sample_rate} Hz.")
            duration_in_seconds = len(self.audio_data) / self.sample_rate
            new_length = int(duration_in_seconds * self.target_sample_rate)
            resampled_audio = np.interp(
                np.linspace(0, len(self.audio_data), new_length),
                np.arange(len(self.audio_data)),
                self.audio_data
            )
            Logger.log("Resampling completed.")
            return resampled_audio
        else:
            Logger.log("Sample rate matches target; no resampling needed.")
            return self.audio_data

    def slice_audio(self):
        """Slices the resampled audio into chunks."""

        samples_per_chunk = int(self.chunk_duration * self.target_sample_rate / 1000)
        chunks_audio_processor = [
            self.resampled_audio[i:i + samples_per_chunk]
            for i in range(0, len(self.resampled_audio), samples_per_chunk)
        ]
        Logger.log(f"Audio sliced into {len(chunks_audio_processor)} chunks.")
        return chunks_audio_processor


class AudioSaver:
    """Saves processed audio chunks to specified output directory."""

    def __init__(self, output_dir_audio_saver="resampled_audio"):
        self.output_dir = Path(output_dir_audio_saver)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        Logger.log(f"Output directory set to '{self.output_dir}'.")

    def save_chunks(self, chunks_audio_saver, base_name="resampled_chunk", format_audio_saver="wav"):
        """Saves each chunk as an individual file."""

        for i, chunk in enumerate(chunks_audio_saver):
            chunk_path = self.output_dir / f"{base_name}_{i}.{format_audio_saver}"
            sf.write(chunk_path, chunk, 16000)  # Save each chunk at 16000 Hz
            Logger.log(f"Resampled chunk {i + 1} saved as {chunk_path}")

