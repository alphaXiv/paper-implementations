import os
import tempfile

from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from pypdf import PdfReader, PdfWriter


def run_mineru(pdf_path: str, page_num: int = 1) -> str:
    output_folder = tempfile.TemporaryDirectory()
    image_output_folder = tempfile.TemporaryDirectory()

    # Initialize writers (same for all PDFs)
    image_writer = FileBasedDataWriter(image_output_folder.name)
    md_writer = FileBasedDataWriter(output_folder.name)

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
        # Read the PDF file bytes
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_to_process)

        # Create dataset instance
        ds = PymuDocDataset(pdf_bytes)

        # Inference: decide whether to run OCR mode based on dataset classification
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        # Generate markdown content; the image directory is the basename of the images output folder
        image_dir_basename = os.path.basename(image_output_folder.name)
        # md_content = pipe_result.get_markdown(image_dir_basename)

        # Dump markdown file
        with tempfile.NamedTemporaryFile("w+", suffix="md") as tf:
            pipe_result.dump_md(md_writer, tf.name, image_dir_basename)
            tf.flush()

            tf.seek(0)
            md_data = tf.read()

        return md_data
    finally:
        # Clean up the temporary file if it was created
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
