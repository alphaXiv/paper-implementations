import json
import os

from anthropic import Anthropic
from prompts import build_openai_silver_data_prompt, claude_response_format_schema

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts.anchor import get_anchor_text


def run_claude(pdf_path: str, page_num: int = 1, model: str = "claude-3-7-sonnet-20250219", temperature: float = 0.1) -> str:
    """
    Convert page of a PDF file to markdown using Claude OCR.
    This function renders the specified page of the PDF to an image, runs OCR on that image,
    and returns the OCR result as a markdown-formatted string.

    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): The page number to process (starting from 1).
        model (str): The Claude model to use.
        temperature (float): The temperature parameter for generation.

    Returns:
        str: The OCR result in markdown format.
    """

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("You must specify an ANTHROPIC_API_KEY")

    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=2048)
    anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=3000,
        temperature=temperature,
        # system=system_prompt,
        tools=claude_response_format_schema(),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                    {
                        "type": "text",
                        "text": f"{build_openai_silver_data_prompt(anchor_text)}. Use the page_response tool to respond. If the propeties are true, then extract the text from them and respond in natural_text.",
                    },
                ],
            }
        ],
    )

    json_sentiment = None
    for content in response.content:
        if content.type == "tool_use" and content.name == "page_response":
            json_sentiment = content.input
            break

    if json_sentiment:
        response = json.dumps(json_sentiment, indent=2)
        return response
