from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Document:
    file_path: str
    metadata: Dict[str, Any]


@dataclass
class Chunk:
    content: str
    metadata: Dict[str, Any]
