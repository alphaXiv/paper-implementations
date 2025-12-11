import os
import tempfile

from mistralai import Mistral
from pypdf import PdfReader, PdfWriter


def run_mistral(pdf_path: str, page_num: int = 1) -> str:
    """
    Convert page of a PDF file to markdown using the mistral OCR api
    https://docs.mistral.ai/capabilities/document/

    Args:
        pdf_path (str): The local path to the PDF file.

    Returns:
        str: The OCR result in markdown format.
    """
    if not os.getenv("MISTRAL_API_KEY"):
        raise SystemExit("You must specify an MISTRAL_API_KEY")

    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)

    if page_num > 0:  # If a specific page is requested
        reader = PdfReader(pdf_path)

        # Check if the requested page exists
        if page_num > len(reader.pages):
            raise ValueError(f"Page {page_num} does not exist in the PDF. PDF has {len(reader.pages)} pages.")

        # Create a new PDF with just the requested page
        writer = PdfWriter()
        # pypdf uses 0-based indexing, so subtract 1 from page_num
        writer.add_page(reader.pages[page_num - 1])

        # Save the extracted page to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temp_file.close()  # Close the file but keep the name

        with open(temp_file.name, "wb") as output_pdf:
            writer.write(output_pdf)

        pdf_to_process = temp_file.name
    else:
        pdf_to_process = pdf_path

    try:
        with open(pdf_to_process, "rb") as pf:
            uploaded_pdf = client.files.upload(
                file={
                    "file_name": os.path.basename(pdf_path),
                    "content": pf,
                },
                purpose="ocr",
            )

        signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

        ocr_response = client.ocr.process(
            model="mistral-ocr-2503",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
        )

        client.files.delete(file_id=uploaded_pdf.id)

        return ocr_response.pages[0].markdown
    finally:
        # Clean up the temporary file if it was created
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
