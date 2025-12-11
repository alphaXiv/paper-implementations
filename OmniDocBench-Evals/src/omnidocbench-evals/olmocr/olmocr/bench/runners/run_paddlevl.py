from threading import Lock

from paddleocr import PaddleOCRVL

# Using docs from here: https://huggingface.co/PaddlePaddle/PaddleOCR-VL
paddle_pipeline = None
paddle_pipeline_lock = Lock()


def run_paddlevl(pdf_path: str, page_num: int = 1, **kwargs) -> str:
    global paddle_pipeline

    with paddle_pipeline_lock:
        if paddle_pipeline is None:
            paddle_pipeline = PaddleOCRVL()

    output = paddle_pipeline.predict(pdf_path)
    result = ""
    for cur_page_0_indexed, res in enumerate(output):
        if cur_page_0_indexed == page_num - 1:
            result = res.markdown["markdown_texts"]

    return result
