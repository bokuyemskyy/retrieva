from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from typing import List, Tuple
from uuid import uuid4

from core.models import Chunk, Modality, Document


def clean_text(text: str) -> str:
    text = "".join(
        ch for ch in text if unicodedata.category(ch) != "Cc" or ch in "\n\t"
    )
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n ", "\n", text)
    return text.strip()


class BaseChunker(ABC):
    def chunk(
        self,
        content: str,
        document: Document,
        modality: Modality,
        metadata: dict | None = None,
    ) -> List[Chunk]:
        content = clean_text(content)

        if not content:
            return []

        metadata = metadata or {}
        chunks: List[Chunk] = []
        chunk_index = 0

        splits = self._split(content)

        for text_segment, start, end in splits:
            segment = text_segment.strip()

            if not segment:
                continue

            chunks.append(
                Chunk(
                    chunk_id=uuid4(),
                    content=segment,
                    document_id=document.document_id,
                    modality=modality,
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "start_char": start,
                        "end_char": end,
                    },
                )
            )
            chunk_index += 1

        return chunks

    @abstractmethod
    def _split(self, text: str) -> List[Tuple[str, int, int]]:
        raise NotImplementedError


class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._step = chunk_size - chunk_overlap

    def _split(self, text: str) -> List[Tuple[str, int, int]]:
        splits = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            raw_window = text[start:end]

            if end < len(text):
                last_space = raw_window.rfind(" ")
                if last_space > 0:
                    end = start + last_space
                    raw_window = text[start:end]

            splits.append((raw_window, start, end))
            start += self._step

        return splits


class SentenceChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 800):
        self.max_chunk_size = max_chunk_size

    def _split(self, text: str) -> List[Tuple[str, int, int]]:
        sentences = []
        for match in re.finditer(r"[^.!?\n]*[.!?\n]+(?:\s|$)+|.+", text):
            sentences.append((match.group(), match.start(), match.end()))

        splits = []
        current_text = ""
        current_start = 0

        for sentence_text, s_start, s_end in sentences:
            if not current_text:
                current_start = s_start

            if (
                len(current_text) + len(sentence_text) > self.max_chunk_size
                and current_text
            ):
                splits.append(
                    (current_text, current_start, current_start + len(current_text))
                )
                current_text = sentence_text
                current_start = s_start
            else:
                current_text += sentence_text

        if current_text:
            splits.append(
                (current_text, current_start, current_start + len(current_text))
            )

        return splits


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " "]

    def _split(self, text: str) -> List[Tuple[str, int, int]]:
        return self._split_recursively(text, 0, self.separators)

    def _split_recursively(
        self, text: str, offset: int, separators: List[str]
    ) -> List[Tuple[str, int, int]]:

        if len(text) <= self.chunk_size:
            return [(text, offset, offset + len(text))]

        active_sep = ""
        for sep in separators:
            if sep in text:
                active_sep = sep
                break

        if not active_sep:
            splits = []
            start = 0
            step = self.chunk_size - self.chunk_overlap
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                splits.append((text[start:end], offset + start, offset + end))
                start += step
            return splits

        parts = text.split(active_sep)
        splits = []

        current_pieces: list[str] = []
        current_length = 0
        current_start = 0
        running_offset = 0

        for i, part in enumerate(parts):
            sep_len = len(active_sep) if i < len(parts) - 1 else 0
            part_len = len(part)

            if current_length + part_len > self.chunk_size and current_pieces:
                chunk_text = active_sep.join(current_pieces)
                chunk_end = current_start + len(chunk_text)
                splits.append((chunk_text, offset + current_start, offset + chunk_end))

                current_pieces = [part]
                current_start = running_offset
                current_length = part_len + sep_len
            else:
                current_pieces.append(part)
                current_length += part_len + sep_len

            running_offset += part_len + sep_len

        if current_pieces:
            chunk_text = active_sep.join(current_pieces)
            splits.append(
                (
                    chunk_text,
                    offset + current_start,
                    offset + current_start + len(chunk_text),
                )
            )

        final_splits = []
        next_separators = (
            separators[separators.index(active_sep) + 1 :]
            if active_sep in separators
            else []
        )

        for chunk_text, start_idx, end_idx in splits:
            if len(chunk_text) > self.chunk_size and next_separators:
                final_splits.extend(
                    self._split_recursively(chunk_text, start_idx, next_separators)
                )
            else:
                final_splits.append((chunk_text, start_idx, end_idx))

        return final_splits
