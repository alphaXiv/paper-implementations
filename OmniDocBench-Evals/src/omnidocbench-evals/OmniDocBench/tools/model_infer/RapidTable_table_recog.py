from pathlib import Path
import json
import os
from PIL import Image
from rapid_table import RapidTable, VisTable
from rapidocr_onnxruntime import RapidOCR
from rapid_table.table_structure.utils import trans_char_ocr_res

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

    table_engine = RapidTable()

    # input_args = RapidTableInput(use_cuda=True)
    # table_engine = RapidTable(input_args)


    # input_args = RapidTableInput(model_type="unitable", use_cuda=True, device="cuda:0")
    # table_engine = RapidTable(input_args)

    ocr_engine = RapidOCR()

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
            if not anno["category_type"] == 'table':
                continue
            # print(anno)
            bbox = poly2bbox(anno['poly'])
            image = img.crop(bbox).convert('RGB') # crop text block
            image.save('table_tmp.jpg')
            ocr_result, _ = ocr_engine('table_tmp.jpg')


            # ocr_result, _ = ocr_engine(img_path, return_word_box=True)
            # ocr_result = trans_char_ocr_res(ocr_result)

            if ocr_result is not None:
                table_results = table_engine(img_path, ocr_result)
                pred_html = table_results.pred_html
            else:
                pred_html = ''


            anno['pred'] = pred_html
        with open('../demo_data/recognition/OmniDocBench_demo_table.jsonl', 'a', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def save_json():

    with open('../demo_data/recognition/OmniDocBench_demo_table.jsonl', 'r') as f:
        lines = f.readlines()
    samples = [json.loads(line) for line in lines]
    with open('../demo_data/recognition/OmniDocBench_demo_table.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
    save_json()