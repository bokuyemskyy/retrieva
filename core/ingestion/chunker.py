from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from uuid import uuid4

from core.models import Chunk, Modality, Document


def clean_text(text: str) -> str:
    text = "".join(
        ch for ch in text if unicodedata.category(ch) != "Cc" or ch in "\n\t"
    )
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class BaseChunker(ABC):
    def chunk(
        self,
        content: str,
        document: Document,
        modality: Modality,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        content = clean_text(content)

        if not content:
            return []

        metadata = metadata or {}
        chunks: list[Chunk] = []
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
    def _split(self, text: str) -> list[tuple[str, int, int]]:
        raise NotImplementedError


class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split(self, text: str) -> list[tuple[str, int, int]]:
        splits = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            if end < len(text):
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            splits.append((text[start:end], start, end))

            if end == len(text):
                break

            start = end - self.chunk_overlap

            if start <= splits[-1][1]:
                start = end

        return splits


class SentenceChunker(BaseChunker):
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split(self, text: str) -> list[tuple[str, int, int]]:
        sentences = [
            (m.group(), m.start(), m.end())
            for m in re.finditer(
                r"[^.!?\n]*[.!?\n]+(?:\s|$)+|.+",
                text,
            )
        ]

        if not sentences:
            return []

        splits: list[tuple[str, int, int]] = []
        i = 0

        while i < len(sentences):
            chunk_start_idx = i
            chunk_sentences = []
            chunk_length = 0

            while i < len(sentences):
                sentence_text, _, _ = sentences[i]
                sentence_len = len(sentence_text)

                if chunk_sentences and (chunk_length + sentence_len > self.chunk_size):
                    break

                if not chunk_sentences and sentence_len > self.chunk_size:
                    chunk_sentences.append(sentences[i])
                    i += 1
                    break

                chunk_sentences.append(sentences[i])
                chunk_length += sentence_len
                i += 1

            chunk_text = "".join(s[0] for s in chunk_sentences)
            start_char = chunk_sentences[0][1]
            end_char = chunk_sentences[-1][2]

            splits.append((chunk_text, start_char, end_char))

            if i >= len(sentences):
                break

            overlap_chars = 0
            overlap_count = 0

            for sentence in reversed(chunk_sentences):
                sentence_len = len(sentence[0])

                if overlap_chars + sentence_len > self.chunk_overlap:
                    break

                overlap_chars += sentence_len
                overlap_count += 1

            next_i = max(chunk_start_idx + 1, i - overlap_count)

            if next_i <= chunk_start_idx:
                next_i = chunk_start_idx + 1

            i = next_i

        return splits


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " "]

    def _split(self, text: str) -> list[tuple[str, int, int]]:
        return self._split_recursively(text, 0, self.separators)

    def _split_recursively(
        self, text: str, offset: int, separators: list[str]
    ) -> list[tuple[str, int, int]]:

        if len(text) <= self.chunk_size:
            return [(text, offset, offset + len(text))]

        active_sep = ""
        for sep in separators:
            if sep in text:
                active_sep = sep
                break

        def _hard_split(
            fallback_text: str, fallback_offset: int
        ) -> list[tuple[str, int, int]]:
            fallback_splits = []
            start = 0
            step = self.chunk_size - self.chunk_overlap
            while start < len(fallback_text):
                end = min(start + self.chunk_size, len(fallback_text))
                fallback_splits.append(
                    (
                        fallback_text[start:end],
                        fallback_offset + start,
                        fallback_offset + end,
                    )
                )
                start += step
            return fallback_splits

        if not active_sep:
            return _hard_split(text, offset)

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
            if len(chunk_text) > self.chunk_size:
                if next_separators:
                    final_splits.extend(
                        self._split_recursively(chunk_text, start_idx, next_separators)
                    )
                else:
                    final_splits.extend(_hard_split(chunk_text, start_idx))
            else:
                final_splits.append((chunk_text, start_idx, end_idx))

        return final_splits
