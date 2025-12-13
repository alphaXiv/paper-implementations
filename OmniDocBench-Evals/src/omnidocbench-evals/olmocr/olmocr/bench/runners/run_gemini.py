import base64
import json
import os
from typing import Literal

from google import genai
from google.genai import types

from olmocr.bench.prompts import (
    build_openai_silver_data_prompt_no_document_anchoring,
)
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts.anchor import get_anchor_text
from olmocr.prompts.prompts import build_openai_silver_data_prompt


def run_gemini(
    pdf_path: str,
    page_num: int = 1,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.1,
    target_longest_image_dim: int = 2048,
    prompt_template: Literal["full", "full_no_document_anchoring", "basic", "finetune"] = "finetune",
    response_template: Literal["plain", "json"] = "json",
) -> str:
    """
    Convert page of a PDF file to markdown using Gemini's vision capabilities.
    This function renders the specified page of the PDF to an image, runs OCR on that image,
    and returns the OCR result as a markdown-formatted string.

    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): The page number to process (starting from 1).
        model (str): The Gemini model to use.
        temperature (float): The temperature parameter for generation.

    Returns:
        str: The OCR result in markdown format.
    """
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("You must specify an GEMINI_API_KEY")

    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=2048)
    anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport")
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    image_part = types.Part(inline_data=types.Blob(mime_type="image/png", data=base64.b64decode(image_base64)))

    if prompt_template == "full":
        text_part = types.Part(text=f"""{build_openai_silver_data_prompt(anchor_text)}""")
    elif prompt_template == "full_no_document_anchoring":
        text_part = types.Part(text=f"""{build_openai_silver_data_prompt_no_document_anchoring(anchor_text)}""")
    else:
        raise NotImplementedError()

    if response_template == "json":
        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=1.0,
            top_k=32,
            max_output_tokens=4096,
            response_mime_type="application/json",
            response_schema=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                required=["primary_language", "is_rotation_valid", "rotation_correction", "is_table", "is_diagram", "natural_text"],
                properties={
                    "primary_language": genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                    "is_rotation_valid": genai.types.Schema(
                        type=genai.types.Type.BOOLEAN,
                    ),
                    "rotation_correction": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        enum=["0", "90", "180", "270"],
                    ),
                    "is_table": genai.types.Schema(
                        type=genai.types.Type.BOOLEAN,
                    ),
                    "is_diagram": genai.types.Schema(
                        type=genai.types.Type.BOOLEAN,
                    ),
                    "natural_text": genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                },
            ),
        )

        response = client.models.generate_content(
            model=f"models/{model}",
            contents=[types.Content(parts=[image_part, text_part])],
            config=generation_config,
        )

        assert len(response.candidates) > 0, "No candidates found"
        assert response.candidates[0].finish_reason == types.FinishReason.STOP, "Finish reason was not STOP, likely a processing error or repetition failure"

        result = response.candidates[0].content.parts[0].text
        parsed = json.loads(result)

        # The json schema is slightly off with gemini vs chatgpt, so we don't verify it
        return parsed["natural_text"]
    else:
        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=1.0,
            top_k=32,
            max_output_tokens=4096,
        )

        response = client.models.generate_content(
            model=f"models/{model}",
            contents=[types.Content(parts=[image_part, text_part])],
            config=generation_config,
        )

        assert len(response.candidates) > 0, "No candidates found"
        assert response.candidates[0].finish_reason == types.FinishReason.STOP, "Finish reason was not STOP, likely a processing error or repetition failure"

        result = response.candidates[0].content.parts[0].text
        return result
