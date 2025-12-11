import os
import subprocess
from typing import List, Union


def convert_image_to_pdf_bytes(image_files: Union[str, List[str]]) -> bytes:
    """
    Convert one or multiple image files to PDF bytes.

    Args:
        image_files: A single image file path (str) or a list of image file paths

    Returns:
        bytes: The PDF content as bytes

    Raises:
        RuntimeError: If the conversion fails
        ValueError: If invalid input is provided
    """
    # Handle different input types
    if isinstance(image_files, str):
        # Single image case
        image_files = [image_files]
    elif not isinstance(image_files, list) or not image_files:
        raise ValueError("image_files must be a non-empty string or list of strings")

    # Validate files exist and are valid image formats
    for image_file in image_files:
        if not os.path.exists(image_file):
            raise ValueError(f"File does not exist: {image_file}")

    try:
        # Run img2pdf with all images as arguments
        result = subprocess.run(["img2pdf"] + image_files, check=True, capture_output=True)

        # Return the stdout content which contains the PDF data
        return result.stdout

    except subprocess.CalledProcessError as e:
        # Raise error with stderr information if the conversion fails
        raise RuntimeError(f"Error converting image(s) to PDF: {e.stderr.decode('utf-8')}")


def is_png(file_path):
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
            return header == b"\x89PNG\r\n\x1a\n"
    except Exception as e:
        print(f"Error: {e}")
        return False


def is_jpeg(file_path):
    try:
        with open(file_path, "rb") as f:
            header = f.read(2)
            return header == b"\xff\xd8"
    except Exception as e:
        print(f"Error: {e}")
        return False
