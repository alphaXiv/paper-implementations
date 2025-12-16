from docling.document_converter import DocumentConverter
import os
from tqdm import tqdm
import pdb

converter = DocumentConverter()

img_path = './OmniDocBench/pdfs'

save_path = '../result/docling'

for img_name in tqdm(os.listdir(img_path)):
    if not img_name.endswith('.pdf'):
        continue
    
    img_name = img_name.strip()

    save_result_path = os.path.join(save_path, img_name[:-4] + '.md')
    
    if os.path.exists(save_result_path):
        continue

    img_path_tmp = os.path.join(img_path, img_name)
    try:
        result = converter.convert(img_path_tmp)
        result_md = result.document.export_to_markdown()
    except:
        print(img_name)
        continue

    with open(save_result_path, 'w', encoding='utf-8') as output_file:
        output_file.write(result_md)