from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from core.ingestion.processor.base_file_processor import BaseFileProcessor
from core.ingestion.chunker import BaseChunker
from core.models import Chunk, Document, Modality

WhisperModel = Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"]


class AudioProcessor(BaseFileProcessor):
    def __init__(
        self,
        chunker: BaseChunker,
        model_size: WhisperModel = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = "en",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model = None
        self.chunker = chunker

    def ingest(self, document: Document) -> List[Chunk]:
        path = Path(document.source_path).resolve()

        if not path.is_file():
            raise FileNotFoundError(document.source_path)

        transcript = self._transcribe(path)

        return self.chunker.chunk(
            content=transcript,
            document=document,
            modality=Modality.AUDIO,
            metadata={
                "whisper_model": self._model_size,
                "language": self._language or "auto",
            },
        )

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise ImportError("AudioProcessor requires faster-whisper") from exc

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
