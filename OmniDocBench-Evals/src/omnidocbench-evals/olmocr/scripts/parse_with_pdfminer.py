from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar


def extract_chars_with_transforms(pdf_path, page_num=0):
    """
    Extract characters with transformation data for a specific page in a PDF.

    Args:
        pdf_path (str): Path to the PDF file
        page_num (int): Page number to extract (0-indexed)
    """
    print(f"Analyzing PDF: {pdf_path}, Page: {page_num + 1}")
    char_count = 0

    # Extract only the specified page
    for i, page_layout in enumerate(extract_pages(pdf_path)):
        if i == page_num:
            print(f"Processing page {page_num + 1}")

            # Recursively process all elements
            def process_element(element, level=0):
                nonlocal char_count
                indent = "  " * level

                if isinstance(element, LTChar):
                    char = element.get_text()
                    matrix = element.matrix
                    font = element.fontname if hasattr(element, "fontname") else "Unknown"
                    size = element.size if hasattr(element, "size") else "Unknown"

                    print(f"{indent}Character: '{char}'")
                    print(f"{indent}Transform Matrix: {matrix}")
                    print(f"{indent}Font: {font}, Size: {size}")
                    print(f"{indent}{'-' * 30}")
                    char_count += 1

                # For container elements, process their children
                if hasattr(element, "_objs"):
                    for obj in element._objs:
                        process_element(obj, level + 1)

            # Process all elements in the page
            for element in page_layout:
                process_element(element)

            break  # Stop after processing the requested page

    print(f"\nTotal characters extracted: {char_count}")

    if char_count == 0:
        print("No characters were extracted. This could mean:")
        print(f"1. Page {page_num + 1} doesn't exist or is empty")
        print("2. The PDF contains scanned images rather than text")
        print("3. The text is embedded in a way PDFMiner can't extract")


# Usage

pdf_path = "/Users/kylel/Downloads/olmOCR_Technical_Report_COLM_2025.pdf"
extract_chars_with_transforms(pdf_path)
