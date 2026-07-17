import re
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.logger import logger

# Optional imports for PDF handling
try:
    import pypdf
except ImportError:
    pypdf = None
    logger.warning("pypdf not installed. PDF parsing will be limited.")

@dataclass
class Segment:
    """
    Represents a semantic segment of a document.
    """
    header: str
    content: str
    level: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Returns the full text representation of the segment."""
        return f"{'#' * self.level} {self.header}\n{self.content}"

class DocumentSegmenter:
    """
    Implements Hierarchical Content Segmentation (Algo 1).
    Parses documents into structured chunks (Header, Content) to preserve semantic context.
    """
    
    def __init__(self, use_marker: bool = True):
        self.use_marker = use_marker

    def parse_file(self, file_path: str) -> List[Segment]:
        """
        Entry point to parse a file (PDF or MD) into segments.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() == '.pdf':
            return self.parse_pdf(str(path))
        elif path.suffix.lower() in ['.md', '.markdown', '.txt']:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.parse_markdown(text)
        else:
            logger.warning(f"Unsupported file format: {path.suffix}. Treating as text.")
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.parse_markdown(text)

    def parse_pdf(self, pdf_path: str) -> List[Segment]:
        """
        Converts PDF to Markdown and segments it.
        Prioritizes 'marker-pdf' for high-fidelity conversion if configured.
        Falls back to pypdf text extraction.
        """
        markdown_text = ""
        
        # Strategy 1: Try marker-pdf (simulated integration as it's a heavy ML dependency)
        # In a real deployment, we would import marker or subprocess it.
        # For this reproduction, we check if we can actually run it, otherwise fallback.
        if self.use_marker:
            try:
                # Placeholder for actual marker-pdf call
                # from marker.convert import convert_single_pdf
                # full_text, images, out_meta = convert_single_pdf(pdf_path, model, metadata)
                # markdown_text = full_text
                
                # Since we can't easily install heavy ML libs in this environment, 
                # we will fallback to pypdf but structure the text to look like markdown headers
                # if possible, or just treat it as raw text.
                logger.info("marker-pdf integration is a placeholder. Falling back to pypdf.")
                pass
            except Exception as e:
                logger.warning(f"Failed to use marker-pdf: {e}")

        # Strategy 2: pypdf Fallback
        if not markdown_text and pypdf:
            try:
                reader = pypdf.PdfReader(pdf_path)
                text_parts = []
                for page in reader.pages:
                    text_parts.append(page.extract_text())
                
                # Simple heuristic to create structure from raw text
                # This is much worse than marker-pdf but allows the code to run without ML deps
                raw_text = "\n".join(text_parts)
                markdown_text = self._heuristic_pdf_to_markdown(raw_text)
            except Exception as e:
                logger.error(f"Failed to parse PDF with pypdf: {e}")
                return []

        return self.parse_markdown(markdown_text)

    def _heuristic_pdf_to_markdown(self, text: str) -> str:
        """
        A simple heuristic to convert raw PDF text to pseudo-markdown.
        It assumes lines that are short and all caps or title case might be headers.
        """
        lines = text.split('\n')
        md_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                md_lines.append("")
                continue
            
            # Heuristic: Short lines (<60 chars) that don't end in punctuation might be headers
            # This is very rough and intended as a fallback.
            if len(stripped) < 60 and not stripped[-1] in ".,:;":
                # Check if it looks like a title
                if stripped.isupper() or stripped.istitle():
                    md_lines.append(f"## {stripped}")
                else:
                    md_lines.append(stripped)
            else:
                md_lines.append(stripped)
        
        return "\n".join(md_lines)

    def parse_markdown(self, text: str) -> List[Segment]:
        """
        Parses Markdown text into hierarchical segments.
        Algorithm:
        1. Identify headers (#, ##, etc.)
        2. Group content under the preceding header.
        3. Maintain a stack or list of segments.
        """
        segments: List[Segment] = []
        lines = text.split('\n')
        
        current_header = "Root"
        current_level = 0
        current_content = []
        
        # Regex to match markdown headers
        header_pattern = re.compile(r'^(#{1,6})\s+(.*)')
        
        for line in lines:
            match = header_pattern.match(line)
            if match:
                # If we have accumulated content for the previous header, save it
                if current_content or current_header != "Root":
                    segments.append(Segment(
                        header=current_header,
                        content="\n".join(current_content).strip(),
                        level=current_level
                    ))
                
                # Start new segment
                hashes, title = match.groups()
                current_level = len(hashes)
                current_header = title.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Append the last segment
        if current_content or current_header != "Root":
            segments.append(Segment(
                header=current_header,
                content="\n".join(current_content).strip(),
                level=current_level
            ))
            
        return segments

if __name__ == "__main__":
    # Simple test
    parser = DocumentSegmenter()
    sample_md = """
# Introduction
This is the intro.

## Background
Some background info.

# Methodology
We used X and Y.
    """
    segs = parser.parse_markdown(sample_md)
    for s in segs:
        print(f"[{s.level}] {s.header}: {s.content[:20]}...")
