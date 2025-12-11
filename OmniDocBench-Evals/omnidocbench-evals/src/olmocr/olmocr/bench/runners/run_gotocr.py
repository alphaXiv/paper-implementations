import base64
import os
import tempfile

import torch
from transformers import AutoModel, AutoTokenizer

from olmocr.data.renderpdf import render_pdf_to_base64png

# Global cache for the model and tokenizer.
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_tokenizer = None


def load_model():
    """
    Load the GOT-OCR model and tokenizer if they haven't been loaded already.
    Returns:
        model: The GOT-OCR model loaded on the appropriate device.
        tokenizer: The corresponding tokenizer.
    """
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("ucaslcl/GOT-OCR2_0", trust_remote_code=True)
        _model = AutoModel.from_pretrained(
            "ucaslcl/GOT-OCR2_0",
            trust_remote_code=True,
            use_safetensors=True,
            revision="979938bf89ccdc949c0131ddd3841e24578a4742",
            pad_token_id=_tokenizer.eos_token_id,
        )
        _model = _model.eval().to(_device)
    return _model, _tokenizer


def run_gotocr(pdf_path: str, page_num: int = 1, ocr_type: str = "ocr") -> str:
    """
    Convert page of a PDF file to markdown using GOT-OCR.

    This function renders the first page of the PDF to an image, runs OCR on that image,
    and returns the OCR result as a markdown-formatted string.

    Args:
        pdf_path (str): The local path to the PDF file.

    Returns:
        str: The OCR result in markdown format.
    """
    # Ensure the model is loaded (cached across calls)
    model, tokenizer = load_model()

    # Convert the first page of the PDF to a base64-encoded PNG image.
    base64image = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=1024)

    # Write the image to a temporary file.
    with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as tmp:
        tmp.write(base64.b64decode(base64image))
        tmp_filename = tmp.name

    # Run GOT-OCR on the saved image.
    result = model.chat(tokenizer, tmp_filename, ocr_type=ocr_type)

    # Clean up the temporary file.
    os.remove(tmp_filename)

    return result
