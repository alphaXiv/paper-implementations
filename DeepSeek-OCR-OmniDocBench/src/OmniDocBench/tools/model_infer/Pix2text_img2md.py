import os
from pathlib import Path
from pix2text import Pix2Text

text_formula_config = dict(
    languages=('en', 'ch_sim'),
    mfd=dict(
        model_path=os.path.expanduser(
            "YOUR MODEL PATH"
        ),
    ),
    formula=dict(
        model_name='mfr-pro',
        model_backend='onnx',
        model_dir=os.path.expanduser(
            "YOUR MODEL PATH"
        ),
    ),
    text=dict(
        rec_model_name='doc-densenet_lite_136-gru',
        rec_model_backend='pytorch',
        rec_model_fp=os.path.expanduser(
            "YOUR MODEL PATH"
        ),
    ),
)

table_config = {
    'model_path': os.path.expanduser(
        "YOUR MODEL PATH"
    )
}

layout_config = {
    'model_type': 'DocXLayoutParser',
    'table_as_image': False,
}

total_config = {
    'layout': layout_config,
    'text_formula': text_formula_config,
    'table': table_config,
}

p2t = Pix2Text.from_config(
    total_configs=total_config,
    enable_formula=True,
    enable_table=True,
)

def batch_process_pdfs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = list(Path(input_dir).glob('*.pdf'))
    for pdf_path in pdf_files:
        try:
            doc = p2t.recognize_pdf(str(pdf_path))
            output_subdir = os.path.join(output_dir, pdf_path.stem)
            doc.to_markdown(output_subdir)
            print(f"Process sucessfully: {pdf_path.name}")
        except Exception as e:
            print(f"Process {pdf_path.name} wrong: {str(e)}")

if __name__ == "__main__":
    input_directory = "/path/to/input/folder"
    output_directory = "/path/to/output/folder"
    batch_process_pdfs(input_directory, output_directory)