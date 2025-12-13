import asyncio
import os
import tempfile
from typing import Literal

from pypdf import PdfReader, PdfWriter


async def run_docling(
    pdf_path: str,
    page_num: int = 1,
    output_format: Literal["markdown"] = "markdown",
    use_smoldocling: bool = False,
) -> str:
    """Run docling CLI on a PDF file and return the results.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to process (1-indexed)
        output_format: Output format (only markdown is supported for CLI version)

    Returns:
        String containing the markdown output
    """
    if output_format != "markdown":
        raise ValueError("Only markdown output format is supported for CLI version")

    # Extract the specific page using pypdf
    pdf_reader = PdfReader(pdf_path)
    pdf_writer = PdfWriter()

    # Convert from 1-indexed to 0-indexed
    zero_based_page_num = page_num - 1

    if zero_based_page_num >= len(pdf_reader.pages) or zero_based_page_num < 0:
        raise ValueError(f"Page number {page_num} is out of bounds for PDF with {len(pdf_reader.pages)} pages")

    # Add the selected page to the writer
    pdf_writer.add_page(pdf_reader.pages[zero_based_page_num])

    # Create temporary files for the single-page PDF and output markdown
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf_file, tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp_md_file:
        tmp_pdf_path = tmp_pdf_file.name
        tmp_md_path = tmp_md_file.name

    try:
        # Write the single-page PDF to the temporary file
        with open(tmp_pdf_path, "wb") as f:
            pdf_writer.write(f)

        # Build the command to run docling on the single-page PDF
        if use_smoldocling:
            cmd = ["docling", tmp_pdf_path, "-o", tmp_md_path]  # Output file
        else:
            cmd = ["docling", "--pipeline", "vlm", "--vlm-model", "smoldocling", tmp_pdf_path, "-o", tmp_md_path]  # Output file

        # Run the command asynchronously
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"docling command failed with return code {proc.returncode}: {error_msg}")

        # Read the results from the temporary markdown file
        with open(tmp_md_path, "r", encoding="utf-8") as f:
            result = f.read()

        return result

    finally:
        # Clean up the temporary files
        for path in [tmp_pdf_path, tmp_md_path]:
            if os.path.exists(path):
                os.unlink(path)
