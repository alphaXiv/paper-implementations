import datetime
import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class PdfOutput:
    path: str
    text: str
    total_pdf_pages: int
    processed_pdf_pages: int

    def mk_dolma_doc(self, **kwargs) -> str:
        metadata = {
            "Source-File": self.path,
            "pdf-pages": self.processed_pdf_pages,
            "pdf-total-pages": self.total_pdf_pages,
            # Kwargs are added as extra metadata
            **kwargs,
        }
        id_ = hashlib.sha1(self.text.encode()).hexdigest()

        dolma_doc = {
            "id": id_,
            "text": self.text,
            "source": "s2pdf",
            "added": datetime.datetime.now().strftime("%Y-%m-%d"),
            "created": datetime.datetime.now().strftime("%Y-%m-%d"),
            "metadata": metadata,
        }

        return json.dumps(dolma_doc)
