from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import List, Optional
from uuid import UUID

from core.models import Chunk


class ObjectStorage:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _workspace_dir(self, workspace: str) -> Path:
        return self.base_dir / workspace

    def save_file(self, workspace: str, document_id: UUID, source_path: str) -> str:
        source = Path(source_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Source file does not exist: {source}")

        workspace_dir = self._workspace_dir(workspace)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        ext = source.suffix.lower()
        destination = workspace_dir / f"{str(document_id)}{ext}"

        shutil.copy2(source, destination)
        return str(destination)

    def delete_file(self, workspace: str, document_id: UUID, extension: str) -> None:
        exact = self._workspace_dir(workspace) / f"{document_id}{extension}"
        exact.unlink(missing_ok=True)

    def save_chunks_cache(
        self, workspace: str, content_hash: str, chunks: List[Chunk]
    ) -> None:
        cache_dir = self._workspace_dir(workspace) / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{content_hash}.pkl"

        with open(cache_file, "wb") as f:
            pickle.dump(chunks, f)

    def load_chunks_cache(
        self, workspace: str, content_hash: str
    ) -> Optional[List[Chunk]]:
        cache_file = self._workspace_dir(workspace) / ".cache" / f"{content_hash}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                cache_file.unlink()
        return None

    def create_workspace(self, workspace: str) -> None:
        workspace_dir = self._workspace_dir(workspace)
        workspace_dir.mkdir(parents=True, exist_ok=True)

    def delete_workspace(self, workspace: str) -> None:
        workspace_dir = self._workspace_dir(workspace)
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)

    def delete_all_workspaces(self) -> None:
        if not self.base_dir.exists():
            return

        for item in self.base_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink(missing_ok=True)
