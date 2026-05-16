from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from .base_file_processor import BaseFileProcessor
from ..chunker import Chunker
from ..image_captioner import ImageCaptioner
from ...model.chunk import Chunk, Modality

WhisperModel = Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"]


class AudioProcessor(BaseFileProcessor):
    """
    Transcribes audio to text.

    Supported file extensions:
    .mp3  .wav  .flac  .ogg  .m4a  .aac  .opus  .webm

    Parameters
    model_size:
        faster-whisper model to load.
    device:
        "cpu" or "cuda".
    compute_type:
        Quantisation level.
    language:
        Language code.
    """

    def __init__(
        self,
        model_size: WhisperModel = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = "en",
        chunker: Optional[Chunker] = None,
        image_processor: Optional[ImageCaptioner] = None,
    ) -> None:
        super().__init__(chunker=chunker, image_processor=image_processor)
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model = None

    def ingest(self, file_path: str) -> List[Chunk]:
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        transcript = self._transcribe(path)

        return self.chunker.chunk(
            text=transcript,
            source_path=str(path),
            modality=Modality.AUDIO,
            base_metadata={
                "filename": path.name,
                "whisper_model": self._model_size,
                "language": self._language or "auto",
            },
        )

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise ImportError("AudioIngestor requires faster-whisper") from exc

        if self._model is None:
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    def _transcribe(self, path: Path) -> str:
        model = self._load_model()
        segments, _info = model.transcribe(
            str(path),
            language=self._language,
            beam_size=5,
        )
        return " ".join(seg.text.strip() for seg in segments)
