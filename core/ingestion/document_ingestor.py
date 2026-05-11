import fitz

from core.ingestion.base_ingestor import BaseIngestor, Document


class DocumentIngestor(BaseIngestor):
    def __init__(self):
        super().__init__()

    def ingest()