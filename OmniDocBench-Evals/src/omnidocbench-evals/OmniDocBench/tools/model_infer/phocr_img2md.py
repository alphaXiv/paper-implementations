import os
from phocr import PHOCR
from pathlib import Path

def process_images_in_folder(input_folder_path: str, output_folder_path: str):
    """
    Recursively processes all image files in a folder and its subfolders,
    saves OCR results as Markdown files in a specified output folder.
    Skips files if a corresponding Markdown file already exists.

    Args:
        input_folder_path (str): The path to the folder containing images.
        output_folder_path (str): The path to the folder where Markdown files will be saved.
    """
    # Initialize OCR engine
    engine = PHOCR()

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)
    print(f"Output folder: {os.path.abspath(output_folder_path)}")

    # Supported image extensions (add more if needed)
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    total_files_to_consider = 0

    # Walk through the input folder and its subfolders
    for root, _, files in os.walk(input_folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_name, file_extension = os.path.splitext(file)

            # Check if the file is an image
            if file_extension.lower() in supported_extensions:
                total_files_to_consider += 1
                output_md_filename = f"{file_name}.md"
                output_md_path = os.path.join(output_folder_path, output_md_filename)

                # Check if the output Markdown file already exists
                if os.path.exists(output_md_path):
                    print(f"Skipping '{file_path}' (Markdown already exists: '{output_md_path}')")
                    skipped_count += 1
                    continue # Move to the next file

                # If the output file doesn't exist, proceed with processing
                print(f"Processing: '{file_path}'")
                try:
                    # Perform OCR on image
                    result = engine(file_path)

                    # Save OCR results to a Markdown file
                    with open(output_md_path, "w", encoding="utf-8") as md_file:
                        md_file.write(result.to_markdown())
                    print(f"  -> Successfully saved Markdown to: '{output_md_path}'")
                    processed_count += 1

                except Exception as e:
                    print(f"  -> ERROR processing '{file_path}': {e}")
                    failed_count += 1
            else:
                print(f"Skipping non-image file: {file_path}")
                pass

    print("\n--- Processing Summary ---")
    print(f"Total image files found to consider: {total_files_to_consider}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (already processed): {skipped_count}")
    print(f"Failed to process: {failed_count}")
    print("--------------------------")

if __name__ == "__main__":
    INPUT_IMAGE_FOLDER = "/path/to/input/folder"  
    OUTPUT_MARKDOWN_FOLDER = "/path/to/output/folder"          

    if not os.path.isdir(INPUT_IMAGE_FOLDER):
        print(f"Error: Input folder '{INPUT_IMAGE_FOLDER}' not found.")
    else:
        process_images_in_folder(INPUT_IMAGE_FOLDER, OUTPUT_MARKDOWN_FOLDER)
        print("\nImage processing script finished.")