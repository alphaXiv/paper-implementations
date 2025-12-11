import httpx

from olmocr.data.renderpdf import render_pdf_to_base64png


async def run_rolmocr(
    pdf_path: str,
    page_num: int = 1,
    server: str = "localhost:30000",
    model: str = "reducto/RolmOCR",
    temperature: float = 0.2,
    target_longest_image_dim: int = 1024,
) -> str:
    """


    Returns:
        str: The OCR result in markdown format.
    """
    # Convert the first page of the PDF to a base64-encoded PNG image.
    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=target_longest_image_dim)

    request = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                    {
                        "type": "text",
                        "text": "Return the plain text representation of this document as if you were reading it naturally.\n",
                    },
                ],
            }
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
        assert (
            choice["finish_reason"] == "stop"
        ), "Response from server did not finish with finish_reason stop as expected, this is probably going to lead to bad data"

        return choice["message"]["content"]
