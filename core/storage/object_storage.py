from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID


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

    def delete_file(self, workspace: str, document_id: UUID) -> None:
        workspace_dir = self._workspace_dir(workspace)
        if not workspace_dir.exists():
            return

        exact = workspace_dir / str(document_id)
        if exact.exists():
            exact.unlink(missing_ok=True)

        for file in workspace_dir.glob(f"{document_id}.*"):
            if file.is_file() or file.is_symlink():
                file.unlink(missing_ok=True)

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
