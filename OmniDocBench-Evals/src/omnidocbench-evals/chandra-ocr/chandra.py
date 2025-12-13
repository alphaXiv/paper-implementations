import os
import time
import requests
from requests.adapters import HTTPAdapter, Retry
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

API_URL = "https://www.datalab.to/api/v1/marker"
API_KEY = os.getenv("DATALAB_API_KEY", "...")

#
# Configure a session with retries, customize retry behavior
#   for your usage needs. Our default rate limit is 200 per minute
#   per account (not per API key).
#
session = requests.Session()
retries = Retry(
    total=20,
    backoff_factor=4,
    status_forcelist=[429],
    allowed_methods=["GET", "POST"],
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("http://", adapter)
session.mount("https://", adapter)


def submit_and_poll_pdf_conversion(
    pdf_path: Path,
    output_format: Optional[str] = 'markdown',
    use_llm: Optional[bool] = False,
    max_polls: int = 300
):
    """
    Submit a PDF for conversion and poll until completion.
    Returns the converted document content.
    """
    def submit_request():
        # Submit initial request
        with open(pdf_path, 'rb') as f:
            form_data = {
                'file': (pdf_path.name, f, 'application/pdf'),
                "max_pages": (None, "123"),
                "force_ocr": (None, False),
                "format_lines": (None, False),
                "paginate": (None, False),
                'output_format': (None, output_format),
                "use_llm": (None, use_llm),
                "strip_existing_ocr": (None, False),
                "disable_image_extraction": (None, False),
                "disable_ocr_math": (None, False),
                "mode": (None, "accurate"),
                "skip_cache": (None, False),
                "save_checkpoint": (None, False)
            }
        
            headers = {"X-Api-Key": API_KEY}
            return session.post(API_URL, headers=headers, files=form_data)

    # Submit the conversion request
    response = submit_request()
    response.raise_for_status()
    data = response.json()
    
    print(f"Submitted: {pdf_path.name}")

    # Poll for completion
    check_url = data["request_check_url"]
    headers = {"X-Api-Key": API_KEY}
    
    for i in range(max_polls):
        response = session.get(check_url, headers=headers)
        check_result = response.json()

        if check_result['status'] == 'complete':
            # Your processing is finished
            converted_document = check_result[output_format]
            print(f"✓ Completed: {pdf_path.name}")
            return converted_document
        elif check_result["status"] == "failed":
            error_msg = f"Failed to convert {pdf_path.name}"
            print(f"✗ {error_msg}")
            raise Exception(error_msg)
        else:
            # Still processing, wait before checking again
            time.sleep(2)
    
    # If we exhausted all polls without completion
    raise TimeoutError(f"Conversion timed out for {pdf_path.name} after {max_polls} polls")


def process_single_pdf(pdf_path: Path, output_dir: Path) -> bool:
    """
    Process a single PDF file and save the result.
    Returns True if successful, False otherwise.
    """
    try:
        # Convert the PDF
        markdown_content = submit_and_poll_pdf_conversion(pdf_path)
        
        # Save the result
        output_filename = pdf_path.stem + '.md'
        output_path = output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"✓ Saved: {output_filename}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {pdf_path.name}: {e}")
        return False


def batch_convert_pdfs(
    document_directory: str = 'images_to_pdf',
    output_directory: str = 'output_results',
    max_workers: int = 3
):
    """
    Batch convert multiple PDFs using threading for parallel processing.
    """
    doc_dir = Path(document_directory)
    output_dir = Path(output_directory)
    
    if not doc_dir.exists():
        print(f"Couldn't find your directory: {document_directory}, exiting early...")
        raise FileNotFoundError(f"Couldn't find {document_directory}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Collect all PDF files
    docs_to_process = list(doc_dir.glob("*.pdf"))
    print(f"Found {len(docs_to_process)} PDFs to convert...")
    
    # Get list of already processed files
    processed_files = set(output_dir.glob("*.md"))
    processed_stems = {f.stem for f in processed_files}
    
    # Filter out already processed PDFs
    docs_to_process = [
        pdf for pdf in docs_to_process 
        if pdf.stem not in processed_stems
    ]
    
    if not docs_to_process:
        print("No new PDFs to process. All files have already been converted.")
        return
    
    print(f"Processing {len(docs_to_process)} new PDFs (skipping already processed)...")

    # Process multiple files at once, up to `max_workers`
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_single_pdf, pdf_path, output_dir): pdf_path.name
            for pdf_path in docs_to_process
        }

        # Use tqdm for progress tracking
        with tqdm(total=len(future_to_file), desc="Converting PDFs") as pbar:
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"✗ Unexpected error processing {filename}: {e}")
                    failed += 1
                finally:
                    pbar.update(1)
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {successful + failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Run the batch conversion
    batch_convert_pdfs(
        document_directory='images_to_pdf',
        output_directory='output_results',
        max_workers=3  # Process 3 PDFs concurrently
    )
