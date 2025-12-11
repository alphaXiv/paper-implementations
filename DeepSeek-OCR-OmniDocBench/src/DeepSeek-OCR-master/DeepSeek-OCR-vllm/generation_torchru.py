import os
import pickle
import argparse
import re
import io
import logging
import threading
import torch.distributed as dist
from datetime import timedelta
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from config import MODEL_PATH, OUTPUT_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, CROP_MODE
from deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.registry import ModelRegistry
from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

# Setup logger
logger = logging.getLogger("generate_ocr")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("generate_ocr.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)


# Set environment variables
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'


def setup_ddp():
    """Initialize the distributed process group for torchrun"""
    # torchrun sets these environment variables
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # Set device for this process
    torch.cuda.set_device(local_rank)

    # Configure for single-machine multi-GPU setup
    if world_size > 1:
        # Set environment variables for NCCL (single machine)
        os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # Use loopback interface
        os.environ['NCCL_IB_DISABLE'] = '1'       # Disable InfiniBand
        os.environ['NCCL_P2P_DISABLE'] = '1'      # Disable P2P
        os.environ['NCCL_SHM_DISABLE'] = '0'      # Enable shared memory
        os.environ['NCCL_DEBUG'] = 'WARN'         # Reduce debug output
        os.environ['NCCL_TIMEOUT'] = '1800000'    # 30 minute timeout in ms

        # Use NCCL for GPU communication
        backend = "nccl"
    else:
        # Single GPU - use gloo
        backend = "gloo"

    try:
        # Initialize the process group
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=1800)  # 30 minute timeout
        )
    except Exception as e:
        print(f"Failed to initialize process group with {backend}: {e}")
        # Fallback to gloo if nccl fails
        if backend == "nccl":
            print("Falling back to gloo backend...")
            dist.init_process_group(
                backend="gloo",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=1800)
            )
        else:
            raise e

    return rank, local_rank, world_size


def cleanup_ddp():
    """Clean up the distributed process group"""
    dist.destroy_process_group()


def load_processing_progress(progress_file):
    """Load progress from previous run to enable resuming"""
    if not os.path.exists(progress_file):
        return None

    try:
        with open(progress_file, 'r') as f:
            lines = f.readlines()

        progress = {}
        for line in lines:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                progress[key.strip()] = value.strip()

        last_file = progress.get('Last file')
        if last_file:
            # Remove .pkl extension for consistency
            if last_file.endswith('.pkl'):
                last_file = last_file[:-4]
            return last_file

    except Exception as e:
        logger.warning(f"Could not read progress file: {e}")

    return None
    """Load progress from previous run to enable resuming"""
    if not os.path.exists(progress_file):
        return None

    try:
        with open(progress_file, 'r') as f:
            lines = f.readlines()

        progress = {}
        for line in lines:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                progress[key.strip()] = value.strip()

        last_file = progress.get('Last file')
        if last_file:
            # Remove .pkl extension for consistency
            if last_file.endswith('.pkl'):
                last_file = last_file[:-4]
            return last_file

    except Exception as e:
        logger.warning(f"Could not read progress file: {e}")

    return None


def load_all_pickle_files(tokenized_dir):
    """Load all pickle file paths from directory"""
    logger.info(f"Loading all pickle file paths from {tokenized_dir}")

    all_pickle_files = [os.path.join(tokenized_dir, f) for f in os.listdir(tokenized_dir)
                       if f.endswith('.pkl')]
    # Sort by modification time (oldest first) instead of alphabetically
    all_pickle_files.sort(key=os.path.getmtime)

    logger.info(f"Found {len(all_pickle_files)} pickle files (sorted by modification time)")
    return all_pickle_files


def load_pickle_batch(file_batch, batch_idx):
    """Load a batch of pickle files using threading for parallel loading"""
    logger.info(f"Loading batch {batch_idx}: {len(file_batch)} files")

    batch_tokenized_data = []
    pdf_info = []
    lock = threading.Lock()

    def load_single_pickle(pickle_file):
        """Load a single pickle file and append to shared lists"""
        try:
            logger.debug(f"Loading {pickle_file}")
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)

            # Handle individual PDF format
            if 'tokenized_data' in data and 'image_paths' in data:
                pdf_name = os.path.splitext(os.path.basename(pickle_file))[0]
                if pdf_name.endswith('.pdf'):
                    pdf_name = pdf_name[:-4]
                num_items = len(data['tokenized_data'])
                num_images = len(data['image_paths'])

                pdf_data = {
                    'name': pdf_name,
                    'num_items': num_items,
                    'num_images': num_images,
                    'image_paths': data['image_paths'],
                    'metadata': data,
                    'tokenized_data': data['tokenized_data']
                }

                with lock:
                    batch_tokenized_data.extend(data['tokenized_data'])
                    pdf_info.append(pdf_data)
            else:
                logger.warning(f"Unexpected pickle file format in {pickle_file}")

        except Exception as e:
            logger.error(f"Error loading {pickle_file}: {e}")

    # Create and start threads for parallel loading
    threads = []
    for pickle_file in file_batch:
        thread = threading.Thread(target=load_single_pickle, args=(pickle_file,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    logger.info(f"Batch {batch_idx} loaded: {len(batch_tokenized_data)} items from {len(pdf_info)} PDFs")
    return batch_tokenized_data, pdf_info


def load_single_tokenized_pdf(tokenized_file):
    """Load tokenized data for a single PDF with saved images"""
    logger.debug(f"Loading tokenized data from {tokenized_file}")
    with open(tokenized_file, 'rb') as f:
        data = pickle.load(f)

    # Load images from saved paths
    original_images = []
    for image_path in data['image_paths']:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            original_images.append(img)
        else:
            raise FileNotFoundError(f"Saved image not found: {image_path}")

    logger.debug(f"Loaded {len(original_images)} images from {tokenized_file}")
    return data['tokenized_data'], data, original_images


def save_single_pdf(pdf_data, pdf_outputs, args_output):
    """Save a single PDF with its results and images using threading for image loading"""
    try:
        pdf_name = pdf_data['name']
        num_items = pdf_data['num_items']
        image_paths = pdf_data['image_paths']
        pdf_metadata = pdf_data['metadata']

        # Load images for this PDF only during saving (lazy loading) using threading
        image_results = []
        lock = threading.Lock()

        def load_single_image(image_path, index):
            """Load a single image and store it at the correct index"""
            try:
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    with lock:
                        image_results.append((index, img))
                else:
                    logger.warning(f"Saved image not found: {image_path}")
                    with lock:
                        image_results.append((index, None))
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {e}")
                with lock:
                    image_results.append((index, None))

        # Start threads for parallel image loading
        image_threads = []
        for idx, image_path in enumerate(image_paths):
            thread = threading.Thread(target=load_single_image, args=(image_path, idx))
            image_threads.append(thread)
            thread.start()

        # Wait for all image loading threads to complete
        for thread in image_threads:
            thread.join()

        # Sort images back to correct order and filter out None values
        image_results.sort(key=lambda x: x[0])  # Sort by index
        pdf_images = [img for idx, img in image_results if img is not None]  # Filter None values

        # Create PDF-specific output directory
        pdf_output_dir = os.path.join(args_output, pdf_name)
        os.makedirs(pdf_output_dir, exist_ok=True)
        os.makedirs(f'{pdf_output_dir}/images', exist_ok=True)

        logger.info(f"Saving results for PDF: {pdf_name} to {pdf_output_dir}")

        # Process and save this PDF's results
        save_pdf_results(pdf_outputs, pdf_metadata, pdf_images, pdf_output_dir)

    except Exception as e:
        logger.error(f"Error saving PDF {pdf_data.get('name', 'unknown')}: {e}")


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None
    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, jdx, output_path):
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()

    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                color_a = color + (20, )

                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{output_path}/images/{jdx}_{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1

                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                    fill=(255, 255, 255, 30))

                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, jdx, output_path):
    result_image = draw_bounding_boxes(image, ref_texts, jdx, output_path)
    return result_image


def pil_to_pdf_img2pdf(pil_images, output_path):
    import img2pdf

    if not pil_images:
        return

    image_bytes_list = []

    for img in pil_images:
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        img_bytes = img_buffer.getvalue()
        image_bytes_list.append(img_bytes)

    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
    except Exception as e:
        print(f"error: {e}")


def save_pdf_results(outputs_list, metadata, original_images, output_path):
    """Save OCR results for a single PDF"""

    logger.debug(f"Saving results for {len(outputs_list)} pages to {output_path}")

    # Process results
    mmd_det_path = os.path.join(output_path, 'ocr_detailed.mmd')
    mmd_path = os.path.join(output_path, 'ocr_content.mmd')
    pdf_out_path = os.path.join(output_path, 'ocr_layout.pdf')

    contents_det = ''
    contents = ''
    draw_images = []
    jdx = 0

    for output, img in zip(outputs_list, original_images):
        content = output.outputs[0].text

        if '<｜end▁of▁sentence｜>' in content:
            content = content.replace('<｜end▁of▁sentence｜>', '')
        else:
            if SKIP_REPEAT:
                continue

        page_num = f'\n<--- Page {jdx + 1} --->'
        contents_det += content + f'\n{page_num}\n'

        image_draw = img.copy()
        matches_ref, matches_images, mathes_other = re_match(content)
        result_image = process_image_with_refs(image_draw, matches_ref, jdx, output_path)

        draw_images.append(result_image)

        for idx, a_match_image in enumerate(matches_images):
            content = content.replace(a_match_image, f'![](images/' + str(jdx) + '_' + str(idx) + '.jpg)\n')

        for idx, a_match_other in enumerate(mathes_other):
            content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')

        contents += content + f'\n{page_num}\n'
        jdx += 1

    # Save outputs
    logger.debug("Saving output files")
    with open(mmd_det_path, 'w', encoding='utf-8') as afile:
        afile.write(contents_det)

    with open(mmd_path, 'w', encoding='utf-8') as afile:
        afile.write(contents)

    pil_to_pdf_img2pdf(draw_images, pdf_out_path)

    logger.debug(f"Saved PDF results to {output_path}")


def generate_ocr_outputs(tokenized_data, metadata, original_images, output_path, llm, sampling_params):
    """Generate OCR outputs from tokenized data"""

    logger.info(f"Starting OCR generation for {len(tokenized_data)} items to {output_path}")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f'{output_path}/images', exist_ok=True)

    # Generate outputs in batches
    logger.debug("Starting VLLM generation")
    outputs_list = llm.generate(tokenized_data, sampling_params=sampling_params)

    # Process results
    mmd_det_path = os.path.join(output_path, 'combined_det.mmd')
    mmd_path = os.path.join(output_path, 'combined.mmd')
    pdf_out_path = os.path.join(output_path, 'combined_layouts.pdf')

    contents_det = ''
    contents = ''
    draw_images = []
    jdx = 0

    logger.debug("Processing generation results")
    for output, img in tqdm(zip(outputs_list, original_images), desc="Processing OCR results", unit="page", total=len(outputs_list), ncols=80):
        content = output.outputs[0].text

        if '<｜end▁of▁sentence｜>' in content:
            content = content.replace('<｜end▁of▁sentence｜>', '')
        else:
            if SKIP_REPEAT:
                continue

        page_num = f'\n<--- Page Split --->'
        contents_det += content + f'\n{page_num}\n'

        image_draw = img.copy()
        matches_ref, matches_images, mathes_other = re_match(content)
        result_image = process_image_with_refs(image_draw, matches_ref, jdx, output_path)

        draw_images.append(result_image)

        for idx, a_match_image in enumerate(matches_images):
            content = content.replace(a_match_image, f'![](images/' + str(jdx) + '_' + str(idx) + '.jpg)\n')

        for idx, a_match_other in enumerate(mathes_other):
            content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')

        contents += content + f'\n{page_num}\n'
        jdx += 1

    # Save outputs
    logger.debug("Saving output files")
    with open(mmd_det_path, 'w', encoding='utf-8') as afile:
        afile.write(contents_det)

    with open(mmd_path, 'w', encoding='utf-8') as afile:
        afile.write(contents)

    pil_to_pdf_img2pdf(draw_images, pdf_out_path)

    logger.info(f"Generated outputs saved to {output_path}")
    print(f"Generated outputs saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate OCR outputs from tokenized data")
    parser.add_argument("--tokenized-dir", "-t", help="Directory containing tokenized data batches")
    parser.add_argument("--tokenized-file", "-f", help="Single tokenized data file")
    parser.add_argument("--original-pdfs", "-p", help="Directory containing original PDF files")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--batch-size", "-b", type=int, default=50, help="Number of pickle files to load per VLLM batch (each PDF gets its own output directory)")

    args = parser.parse_args()

    # Check if running with torchrun (distributed)
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

    if is_distributed:
        # torchrun mode - each process handles its portion
        rank, local_rank, world_size = setup_ddp()
        ddp_worker(rank, local_rank, world_size, args)
        cleanup_ddp()
    else:
        # Single process mode
        ddp_worker(0, 0, 1, args)


def ddp_worker(rank, local_rank, world_size, args):
    """Worker function for distributed processing with torchrun"""

    # Set up logging - only rank 0 logs
    if rank == 0:
        logger.info("Starting OCR generation process")
        logger.info(f"Arguments: tokenized_dir={args.tokenized_dir}, tokenized_file={args.tokenized_file}, output={args.output}, batch_size={args.batch_size}")
        logger.info(f"Distributed setup: rank={rank}, local_rank={local_rank}, world_size={world_size}")

    # Initialize VLLM model
    if rank == 0:
        logger.debug("Initializing VLLM model")
    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=MAX_CONCURRENCY,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        disable_mm_preprocessor_cache=True,
        dtype='bfloat16'
    )

    logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822})]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    # Load and process tokenized data in batches
    if rank == 0:
        logger.debug("Starting batch processing of tokenized data")

    if args.tokenized_file:
        # Single file processing - only rank 0 handles this
        if rank == 0:
            tokenized_data, metadata, original_images = load_single_tokenized_pdf(args.tokenized_file)
            logger.info(f"Loaded {len(tokenized_data)} tokenized items and {len(original_images)} images")

            # Process single file
            pdf_name = os.path.splitext(os.path.basename(args.tokenized_file))[0]
            # Remove .pdf extension if present for cleaner folder names
            if pdf_name.endswith('.pdf'):
                pdf_name = pdf_name[:-4]
            pdf_output_dir = os.path.join(args.output, pdf_name)
            os.makedirs(pdf_output_dir, exist_ok=True)
            os.makedirs(f'{pdf_output_dir}/images', exist_ok=True)

            # Generate outputs
            logger.debug("Starting VLLM generation")
            outputs_list = llm.generate(tokenized_data, sampling_params=sampling_params)

            # Save results
            save_pdf_results(outputs_list, metadata, original_images, pdf_output_dir)

    elif args.tokenized_dir:
        all_pickle_files = load_all_pickle_files(args.tokenized_dir)

        if not all_pickle_files:
            if rank == 0:
                logger.warning("No files to process. Exiting.")
            return

        if rank == 0:
            logger.info(f"Processing {len(all_pickle_files)} pickle files in batches of {args.batch_size}")

        # Create progress tracking file (initialize if it doesn't exist)
        progress_file = os.path.join(args.output, "processing_progress.txt")
        os.makedirs(args.output, exist_ok=True)

        # Initialize progress file if it doesn't exist (first run) - only rank 0
        if rank == 0 and not os.path.exists(progress_file):
            with open(progress_file, 'w') as f:
                f.write("Processed batches: 0\n")
                f.write("Processed files: 0\n")
                f.write("Last batch: 0\n")
                f.write("Last file: none\n")
            logger.info("Initialized new progress file for first run")

        # Synchronize all processes
        dist.barrier()

        # Divide files among processes
        files_per_process = (len(all_pickle_files) + world_size - 1) // world_size
        start_idx = rank * files_per_process
        end_idx = min(start_idx + files_per_process, len(all_pickle_files))
        process_files = all_pickle_files[start_idx:end_idx]

        if rank == 0:
            logger.info(f"Distributed workload: {world_size} processes, each handling ~{files_per_process} files")
            logger.info(f"Rank {rank}: processing files {start_idx} to {end_idx-1} ({len(process_files)} files)")

        # Process this process's assigned files
        file_batch_size = args.batch_size
        total_file_batches = (len(process_files) + file_batch_size - 1) // file_batch_size

        processed_batches = 0
        total_files_processed = 0

        for batch_idx in range(total_file_batches):
            file_batch_start = batch_idx * file_batch_size
            file_batch_end = min((batch_idx + 1) * file_batch_size, len(process_files))
            file_batch = process_files[file_batch_start:file_batch_end]

            if rank == 0:
                logger.info(f"Rank {rank} - Processing batch {batch_idx + 1}/{total_file_batches}: files {file_batch_start + 1}-{file_batch_end} ({len(file_batch)} files)")

            # Load this batch of pickle files
            batch_tokenized_data, pdf_info = load_pickle_batch(file_batch, batch_idx + 1)

            if not batch_tokenized_data:
                if rank == 0:
                    logger.warning(f"No valid data in batch {batch_idx + 1}, skipping")
                continue

            # Process this batch through VLLM
            if rank == 0:
                logger.debug(f"Rank {rank} - Starting VLLM generation")
            outputs_list = llm.generate(batch_tokenized_data, sampling_params=sampling_params)

            # Split results back by PDF and save each PDF separately using threading
            item_offset = 0
            pdf_save_threads = []

            for pdf_data in pdf_info:
                pdf_name = pdf_data['name']
                num_items = pdf_data['num_items']

                # Extract this PDF's results from the batch
                pdf_outputs = outputs_list[item_offset:item_offset + num_items]

                # Start a thread to save this PDF
                save_thread = threading.Thread(
                    target=save_single_pdf,
                    args=(pdf_data, pdf_outputs, args.output)
                )
                pdf_save_threads.append(save_thread)
                save_thread.start()

                item_offset += num_items

            # Wait for all PDF saving threads to complete
            for thread in pdf_save_threads:
                thread.join()

            # Update progress
            processed_batches += 1
            total_files_processed += len(pdf_info)

            # Update global progress file - only rank 0 writes
            if rank == 0:
                with open(progress_file, 'w') as f:
                    f.write(f"Processed batches: {processed_batches}/{total_file_batches}\n")
                    f.write(f"Processed files: {total_files_processed}/{len(process_files)}\n")
                    f.write(f"Last batch: {batch_idx + 1}\n")
                    f.write(f"Last file: {os.path.basename(file_batch[-1]) if file_batch else 'none'}\n")

                logger.info(f"Rank {rank} - Completed saving {len(pdf_info)} PDFs from batch {batch_idx + 1} (Total: {total_files_processed}/{len(process_files)} files)")

        if rank == 0:
            logger.info(f"Rank {rank} - Processing complete! Total files processed: {total_files_processed}")

    else:
        if rank == 0:
            logger.error("Either --tokenized-file or --tokenized-dir must be provided")
        return

    if rank == 0:
        logger.info("OCR generation process completed")


if __name__ == "__main__":
    main()