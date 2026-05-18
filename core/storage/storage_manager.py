import shutil
from pathlib import Path
from uuid import uuid4


class StorageManager:
    def __init__(self, base_dir: str, workspace: str):
        self.workspace = workspace
        self.base_dir = Path(base_dir) / workspace
        self.files_dir = self.base_dir / "files"

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)

    def save_file(self, source_path: str, original_filename: str) -> tuple[str, str]:
        document_id = str(uuid4())
        ext = Path(original_filename).suffix

        stored_path = self.files_dir / f"{document_id}{ext}"
        shutil.copy2(source_path, stored_path)

        return document_id, str(stored_path)

    def clear_all_files(self) -> None:
        for item in self.files_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    def delete_workspace_directory(self) -> None:
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
