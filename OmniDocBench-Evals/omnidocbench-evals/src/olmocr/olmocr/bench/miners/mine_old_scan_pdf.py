import json
import os
import time

import requests
from bs4 import BeautifulSoup


def process_jsonl(jsonl_file, output_dir="downloaded_images"):
    """
    Process each line in the JSONL file and download the corresponding images.

    Args:
        jsonl_file (str): Path to the JSONL file
        output_dir (str): Directory to save downloaded images
    """

    os.makedirs(output_dir, exist_ok=True)
    with open(jsonl_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                pdf_path = data.get("pdf", "")
                pdf_filename = os.path.basename(pdf_path)
                url = data.get("url", "")
                if not url:
                    print(f"Line {line_num}: Missing URL, skipping...")
                    continue

                print(f"Processing line {line_num}: {pdf_filename} - {url}")
                image_path = download_image(url, pdf_filename, output_dir)

                if image_path:
                    print(f"Successfully downloaded: {image_path}")
                else:
                    print(f"Failed to download image for {pdf_filename}")
                time.sleep(1)

            except json.JSONDecodeError:
                print(f"Line {line_num}: Invalid JSON, skipping...")
            except Exception as e:
                print(f"Line {line_num}: Error processing - {str(e)}")


def download_image(url, output_filename, output_dir):
    """
    Download the highest resolution JPEG (before JPEG2000) from the Library of Congress URL.

    Args:
        url (str): The initial URL from the JSONL file
        output_filename (str): Filename for the downloaded image
        output_dir (str): Directory to save the image

    Returns:
        str or None: Path to the downloaded image if successful, None otherwise
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        loc_link = soup.find("a", class_="btn btn-outline-primary text-nowrap", title="View the original source for this item in a new tab")
        if not loc_link:
            print(f"Could not find 'View on www.loc.gov' link on {url}")
            return None

        loc_url = loc_link["href"]
        print(f"Found LOC URL: {loc_url}")
        response = requests.get(loc_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        download_select = soup.find("select", id="download")
        if not download_select:
            print(f"Could not find download options on {loc_url}")
            return None

        jpeg_options = [option for option in download_select.find_all("option") if "JPEG" in option.text and "JPEG2000" not in option.text]

        if not jpeg_options:
            print(f"No JPEG options found on {loc_url}")
            return None

        highest_jpeg = jpeg_options[-1]
        image_url = highest_jpeg["value"]

        print(f"Found highest resolution JPEG: {image_url}")
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        if not output_filename.lower().endswith((".jpg", ".jpeg")):
            output_filename = f"{os.path.splitext(output_filename)[0]}.jpg"

        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return output_path

    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download images from Library of Congress based on JSONL file")
    parser.add_argument("jsonl_file", help="Path to the JSONL file containing URLs")
    parser.add_argument("--output", "-o", default="downloaded_images", help="Directory to save downloaded images (default: downloaded_images)")

    args = parser.parse_args()

    print(f"Processing JSONL file: {args.jsonl_file}")
    process_jsonl(args.jsonl_file, args.output)
    print("Processing complete!")


if __name__ == "__main__":
    main()
