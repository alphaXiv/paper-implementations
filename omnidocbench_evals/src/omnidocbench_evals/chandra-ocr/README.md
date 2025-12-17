# Chandra OCR

This folder contains the `chandra.py` script and supporting files for running a small OCR/processing workflow used in the DeepSeek-OCR project.

What it is
- `chandra.py`: a utility/script for local OCR-related tasks (detection/recognition or dataset preprocessing). The script is lightweight and intended for quick experiments using chandra-ocr .

We used the marker/ api with mode as balanced/accurate according to the docs of datalab where chandra model is hosted for public use.

Quick usage
1. install chandra-ocr package

   ```bash
   pip install chandra-ocr
   ```

2. Run the script (example):

   ```bash
   python chandra.py
   ```

   Replace arguments with the ones `chandra.py` expects — check the top of the file for specific options.

Notes
- This README is intentionally brief. For full project-level setup and evaluation instructions, see the repository root `README.md` and `DeepSeek-OCR-master/README.md`.
- If `chandra.py` requires specific model files or data paths, place them in this folder or update the script's configuration accordingly.

Contact
- Open an issue or contact the maintainers in the main repo for questions or to report problems.
