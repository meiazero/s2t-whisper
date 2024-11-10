from pathlib import Path
from typing import Optional

from src.slicing_audio import AudioLoader, AudioProcessor
from src.whisper import Transcriber, FileManager, WhisperModelConfig


class AudioTranscriptionPipeline:
    """Pipeline to handle audio processing and transcription."""

    def __init__(
            self,
            model_id: str = "openai/whisper-large-v2",
            target_sample_rate: int = 16000
    ):
        self.target_sample_rate = target_sample_rate
        self.model_config = WhisperModelConfig(model_id=model_id)
        self.transcriber = Transcriber(config=self.model_config)
        self.file_manager = FileManager()

    def process_audio(
            self,
            audio_path: Path | str,
            output_base_name: Optional[str] = None
    ) -> str:
        """
        Process and transcribe a short audio file.

        Args:
            audio_path: Path to the audio file
            output_base_name: Base name for the output transcription file

        Returns:
            The transcribed text
        """
        # Step 1: Load the audio file
        loader = AudioLoader(file_path=audio_path)

        # Step 2: Process the audio
        processor = AudioProcessor(
            audio_data=loader.audio,
            sample_rate=loader.sample_rate,
            target_sample_rate=self.target_sample_rate
        )

        # Step 3: Resample the audio
        processed_audio = processor.process()

        # Step 4: Transcribe the audio
        transcription = self.transcriber.transcribe(
            audio=processed_audio,
            sample_rate=self.target_sample_rate
        )

        # Step 5: Save the transcription
        if output_base_name:
            self.file_manager.save_transcription(
                transcription=transcription,
                base_name=output_base_name
            )

        return transcription
