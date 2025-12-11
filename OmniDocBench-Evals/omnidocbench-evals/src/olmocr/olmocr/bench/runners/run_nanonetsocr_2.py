import base64
import subprocess

import httpx

from olmocr.data.renderpdf import get_pdf_media_box_width_height


# Logic to set min size from here: https://github.com/NanoNets/Nanonets-OCR2/blob/main/Nanonets-OCR2-Cookbook/image2md.ipynb
def render_pdf_to_base64png_min_short_size(local_pdf_path: str, page_num: int, target_shortest_dim: int = 2048) -> str:
    shortest_dim = min(get_pdf_media_box_width_height(local_pdf_path, page_num))

    # Convert PDF page to PNG using pdftoppm
    pdftoppm_result = subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            "-r",
            str(target_shortest_dim * 72 / shortest_dim),  # 72 pixels per point is the conversion factor
            local_pdf_path,
        ],
        timeout=120,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert pdftoppm_result.returncode == 0, pdftoppm_result.stderr
    return base64.b64encode(pdftoppm_result.stdout).decode("utf-8")


async def run_server(
    pdf_path: str,
    page_num: int = 1,
    server: str = "localhost:30000",
    model: str = "nanonets/Nanonets-OCR2-3B",
    temperature: float = 0.0,
    page_dimensions: int = 1280,
) -> str:
    """
    Convert page of a PDF file to markdown by calling a request
    running against an openai compatible server.

    You can use this for running against vllm, sglang, servers
    as well as mixing and matching different model's.

    It will only make one direct request, with no retries or error checking.

    Returns:
        str: The OCR result in markdown format.
    """
    # Convert the first page of the PDF to a base64-encoded PNG image.
    image_base64 = render_pdf_to_base64png_min_short_size(pdf_path, page_num=page_num, target_shortest_dim=page_dimensions)

    # Now use th
    prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""

    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }

    # Make request and get response using httpx
    url = f"http://{server}/v1/chat/completions"

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(url, json=request)

        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        return choice["message"]["content"]
