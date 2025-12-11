import os

from openai import OpenAI
from runners.run_chatgpt import run_chatgpt
from runners.run_gemini import run_gemini

from olmocr.data.renderpdf import render_pdf_to_base64png


def build_find_difference_prompt(base_text: str) -> str:
    return (
        f"Below is an image of a document page, along with raw textual content previously extracted using different models."
        f"Your goal is to carefully identify the differences between the extracted texts from both models and determine which one is more accurate by comparing them with the image."
        f"Only return the differences and specify which model extracted the text with higher accuracy.\n"
        f"Do not hallucinate.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    )


def combined_output(pdf_path: str) -> str:
    chatgpt_output = run_chatgpt(pdf_path)
    gemini_output = run_gemini(pdf_path)
    return f"ChatGPT OUTPUT: \n" f"{chatgpt_output}\n\n" f"Gemini OUTPUT: \n" f"{gemini_output}"


def run_difference(pdf_path: str, page_num: int = 1, model: str = "gpt-4o-2024-08-06", temperature: float = 0.1) -> str:
    """
    Convert page of a PDF file to markdown using GPT.

    This function renders the first page of the PDF to an image, runs OCR on that image,
    and returns the OCR result as a markdown-formatted string.

    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): Which page from document to pass.
        model (str): Model used to process.
        Temperature (float): Temperature used while utilizing the model.

    Returns:
        str: The result in markdown format.
    """
    # Convert the first page of the PDF to a base64-encoded PNG image.
    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=2048)
    anchor_text = combined_output(pdf_path)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_find_difference_prompt(anchor_text)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        temperature=temperature,
        max_tokens=3000,
    )

    raw_response = response.choices[0].message.content

    return raw_response
