import os
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

def process_folder(input_folder, output_folder, max_new_tokens=4096):
    model_path = "nanonets/Nanonets-OCR-s"
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, 
        torch_dtype="auto", 
        device_map="auto", 
        attn_implementation="flash_attention_2"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            try:
                image_path = os.path.join(input_folder, filename)
                print(f"Processing: {image_path}")
                
                result = ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens)
                
                output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.md')
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result)
                
                print(f"Saved result to: {output_path}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

def ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens=4096):
    prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
    image = Image.open(image_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]


input_folder = "/path/to/input/folder"
output_folder = "/path/to/output/folder"

process_folder(input_folder, output_folder, max_new_tokens=15000)