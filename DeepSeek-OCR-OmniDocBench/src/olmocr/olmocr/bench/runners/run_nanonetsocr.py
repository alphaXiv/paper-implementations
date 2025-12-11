import base64
import os
import re
import tempfile

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from olmocr.data.renderpdf import render_pdf_to_base64png

_model = None
_tokenizer = None
_processor = None
_device = None


def load_model(model_path: str = "nanonets/Nanonets-OCR-s"):
    global _model, _tokenizer, _processor, _device

    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            # attn_implementation="flash_attention_2"
        )
        _model.eval()
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _processor = AutoProcessor.from_pretrained(model_path)

    return _model, _tokenizer, _processor


async def run_nanonetsocr(pdf_path: str, page_num: int = 1, model_path: str = "nanonets/Nanonets-OCR-s", max_new_tokens: int = 4096, **kwargs) -> str:
    """
    Convert page of a PDF file to markdown using NANONETS-OCR.

    This function renders the first page of the PDF to an image, runs OCR on that image,
    and returns the OCR result as a markdown-formatted string.

    Args:
        pdf_path (str): The local path to the PDF file.

    Returns:
        str: The OCR result in markdown format.
    """

    model, tokenizer, processor = load_model(model_path)

    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=1024)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image_data = base64.b64decode(image_base64)
        temp_file.write(image_data)
        temp_image_path = temp_file.name

    try:
        image = Image.open(temp_image_path)
        prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{temp_image_path}"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt", use_fast=True)
        inputs = inputs.to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        cleaned_text = re.sub(r"<page_number>\d+</page_number>", "", output_text[0])

        return cleaned_text

    finally:
        try:
            os.unlink(temp_image_path)
        except Exception as e:
            print(f"Warning: Failed to remove temporary file {temp_image_path}: {e}")
