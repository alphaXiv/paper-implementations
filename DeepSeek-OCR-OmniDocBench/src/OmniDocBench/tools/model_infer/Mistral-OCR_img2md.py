from mistralai import Mistral, DocumentURLChunk, OCRResponse
from pathlib import Path
import time
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ocr_conversion.log"),
        logging.StreamHandler()
    ]
)

api_key = "YOUR API KEY"
client = Mistral(api_key=api_key)

pdf_folder = Path(r"YOUR PDF Folder")
output_folder = Path(r"YOUR PDF Folder")

output_folder.mkdir(parents=True, exist_ok=True)

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    markdowns = []
    for page in ocr_response.pages:
        markdowns.append(page.markdown)
    
    return "\n\n".join(markdowns)

def process_pdf(pdf_path, output_path):
    try:
        logging.info(f"Process file: {pdf_path.name}")
        
        uploaded_file = client.files.upload(
            file={
                "file_name": pdf_path.stem,
                "content": pdf_path.read_bytes(),
            },
            purpose="ocr",
        )
        
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        
        pdf_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url), 
            model="mistral-ocr-latest", 
            include_image_base64=False
        )
        
        combined_markdown = get_combined_markdown(pdf_response)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_markdown)
            
        logging.info(f"Sucessfully : {pdf_path.name} -> {output_path.name}")
        return True
        
    except Exception as e:
        logging.error(f"Process {pdf_path.name} wrong: {str(e)}")
        return False

def main():
    pdf_files = list(pdf_folder.glob("*.pdf"))
    total_files = len(pdf_files)
    logging.info(f"Finf {total_files}  PDF files")
    
    successful = 0
    failed = 0
    
    for pdf_file in tqdm(pdf_files, desc="Process PDF file"):
        output_file = output_folder / f"{pdf_file.stem}.md"
        
        if output_file.exists():
            logging.info(f"Skip finished file: {pdf_file.name}")
            successful += 1
            continue
            
        if process_pdf(pdf_file, output_file):
            successful += 1
        else:
            failed += 1
            
        time.sleep(1)
    
    logging.info(f"Process done! Sucess: {successful}, failure: {failed}, total: {total_files}")

if __name__ == "__main__":
    main()