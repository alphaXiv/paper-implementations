import os
import json
import numpy

from PIL import Image, ImageOps
from paddleocr import PaddleOCR, draw_ocr

def test_paddle(img: Image, lan: str ):
    if lan == 'text_simplified_chinese':
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    elif lan == 'text_english':
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
    else:
        ocr = PaddleOCR(use_angle_cls=True)

    img_add_border = add_white_border(img)
    img_ndarray = numpy.array(img_add_border)
    result = ocr.ocr(img_ndarray, cls=True)
    text = ''
    for idx in range(len(result)):
        res = result[idx]
        if not res:
            continue
        for line in res:
            t = line[1][0]
            print(t)
            text += t
    return text

def add_white_border(img: Image):
    border_width = 50
    border_color = (255, 255, 255)  # 白色
    img_with_border = ImageOps.expand(img, border=border_width, fill=border_color)
    return img_with_border


def poly2bbox(poly):
    L = poly[0]
    U = poly[1]
    R = poly[2]
    D = poly[5]
    L, R = min(L, R), max(L, R)
    U, D = min(U, D), max(U, D)
    bbox = [L, U, R, D]
    return bbox

def main():
    with open('../demo_data/omnidocbench_demo/OmniDocBench_demo.json', 'r') as f:
        samples = json.load(f)
    for sample in samples:
        img_name = os.path.basename(sample['page_info']['image_path'])
        img_path = os.path.join('../demo_data/omnidocbench_demo/images', img_name)
        img = Image.open(img_path)
        if not os.path.exists(img_path):
            print('No exist: ', img_name)
            continue
        for i, anno in enumerate(sample['layout_dets']):
            if not anno.get('text'):
                continue
            print(anno)
            lan = anno['attribute'].get('text_language', 'mixed')
            bbox = poly2bbox(anno['poly'])
            image = img.crop(bbox).convert('RGB') # crop text block
            outputs = test_paddle(image, lan) # !!!! String text block的文本内容

            anno['pred'] = outputs
        with open('../demo_data/recognition/OmniDocBench_demo_text_ocr.jsonl', 'a', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def save_json():
    # 文本OCR质检：gpt-4o/internvl jsonl2json
    with open('../demo_data/recognition/OmniDocBench_demo_text_ocr.jsonl', 'r') as f:
        lines = f.readlines()
    samples = [json.loads(line) for line in lines]
    with open('../demo_data/recognition/OmniDocBench_demo_text_ocr.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
    save_json()