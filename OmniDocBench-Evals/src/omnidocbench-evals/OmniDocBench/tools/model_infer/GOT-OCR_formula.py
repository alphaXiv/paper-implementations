import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from GOT.model import *
from GOT.utils.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO
from GOT.model.plug.blip_process import BlipImageEvalProcessor

from transformers import TextStreamer
import re
from GOT.demo.process_results import punctuation_dict, svg_to_html
import string
import json

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


 
translation_table = str.maketrans(punctuation_dict)

def poly2bbox(poly):
    L = poly[0]
    U = poly[1]
    R = poly[2]
    D = poly[5]
    L, R = min(L, R), max(L, R)
    U, D = min(U, D), max(U, D)
    bbox = [L, U, R, D]
    return bbox

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()

    model.to(device='cuda',  dtype=torch.bfloat16)


    # TODO vary old codes, NEED del 
    image_processor = BlipImageEvalProcessor(image_size=1024)

    image_processor_high =  BlipImageEvalProcessor(image_size=1024)

    use_im_start_end = True

    image_token_len = 256

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
            if anno['category_type'] != 'equation_isolated':
                continue
            
            bbox = poly2bbox(anno['poly'])
            image = img.crop(bbox).convert('RGB')

            w, h = image.size
            # print(image.size)
            
            # qs = 'OCR: '
            qs = 'OCR with format: '

            if use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs 
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv_mode = "mpt"
            args.conv_mode = conv_mode

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # print(prompt)


            inputs = tokenizer([prompt])


            # vary old codes, no use
            image_1 = image.copy()
            image_tensor = image_processor(image)


            image_tensor_1 = image_processor_high(image_1)


            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    input_ids,
                    images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
                    do_sample=False,
                    num_beams = 1,
                    no_repeat_ngram_size = 20,
                    streamer=streamer,
                    max_new_tokens=4096,
                    stopping_criteria=[stopping_criteria]
                    )
                
                outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()
                # print('outputs: ', outputs)

            anno['pred'] = outputs

        with open('../demo_data/recognition/OmniDocBench_demo_GOT_formula.jsonl', 'a', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def save_json():
    with open('../demo_data/recognition/OmniDocBench_demo_GOT_formula.jsonl', 'r') as f:
        lines = f.readlines()
    samples = [json.loads(line) for line in lines]
    with open('../demo_data/recognition/OmniDocBench_demo_GOT_formula.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="./checkpoints/GOT_weights")
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--type", type=str, default='ocr')
    # parser.add_argument("--box", type=str, default= '')
    # parser.add_argument("--color", type=str, default= '')
    # parser.add_argument("--render", action='store_true')
    args = parser.parse_args()

    eval_model(args)
    save_json()