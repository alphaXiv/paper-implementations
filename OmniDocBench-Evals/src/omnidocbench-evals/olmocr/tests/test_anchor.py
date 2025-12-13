import base64
import glob
import io
import json
import os
import re
import tempfile
import unittest

from pypdf import PdfReader

from olmocr.data.renderpdf import (
    get_pdf_media_box_width_height,
    render_pdf_to_base64png,
)
from olmocr.image_utils import convert_image_to_pdf_bytes
from olmocr.prompts.anchor import _linearize_pdf_report, _pdf_report, get_anchor_text


class AnchorTest(unittest.TestCase):
    def testExtractText(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "some_ocr1.pdf")
        reader = PdfReader(local_pdf_path)
        page = reader.pages[0]

        def visitor_body(text, cm, tm, font_dict, font_size):
            print(repr(text), cm, tm, font_size)

        def visitor_op(op, args, cm, tm):
            # print(op, args, cm, tm)
            pass

        page.extract_text(visitor_text=visitor_body, visitor_operand_before=visitor_op)

    def testAnchorBase(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "pdftotext_two_column_issue.pdf")

        report = _pdf_report(local_pdf_path, 2)

        print(report)

        print(get_anchor_text(local_pdf_path, 2, pdf_engine="pdfreport"))

    def testAnchorImage(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "some_ocr1.pdf")

        report = _pdf_report(local_pdf_path, 1)

        print(report)

        print(get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport"))

    def testSmallPage(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "small_page_size.pdf")

        report = _pdf_report(local_pdf_path, 1)

        print(report)

        print(get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport"))

    def testBadUTFSurrogatePairsGeneration(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "badlines.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 4, pdf_engine="pdfreport")

        jsondata = json.dumps({"text": anchor_text})

        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.json as paj

        buffer = io.BytesIO(jsondata.encode("utf-8"))
        paj.read_json(buffer, read_options=paj.ReadOptions(use_threads=False, block_size=len(jsondata)))

    def testLargePromptHint1(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "large_prompt_hint1.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 4, pdf_engine="pdfreport")

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 1000)

    def testLargePromptHint2(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "large_prompt_hint2.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 2, pdf_engine="pdfreport")

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 4000)

    def testLargePromptHint3(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "large_prompt_hint3.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 2, pdf_engine="pdfreport")

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 4000)

    def testNewsPaperPromptHint(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "newspaper.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport")

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 4000)

    def testTobaccoPaperMissingParagraphs(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "tobacco_missed_tokens_pg1.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport")

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 4000)

    def testAnchorOtherLengths(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "tobacco_missed_tokens_pg1.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport", target_length=2000)

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 2000)

        anchor_text = get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport", target_length=6000)

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 6000)

    def testFailingAnchor(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "failing_anchor_pg4.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 4, pdf_engine="pdfreport")

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 4000)

    def testEmptyAnchor(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "tobacco_missed_tokens_pg1.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport", target_length=0)

        self.assertEqual(anchor_text.strip(), "Page dimensions: 612.0x792.0")

    def testEmptyAnchorMatchesImageAnchor(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "edgar.pdf")

        orig_anchor = get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport")
        print(orig_anchor)

        lenneg1_anchor = get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport", target_length=-1)
        print(lenneg1_anchor)

        base64_png = render_pdf_to_base64png(local_pdf_path, 1, target_longest_image_dim=1024)

        # Decode base64 and save to temporary file
        temp_img = tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False)
        temp_img.write(base64.b64decode(base64_png))
        temp_img.close()

        # Update pdf_path to the new file
        img_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "edgar_image.pdf")

        image_only_anchor = get_anchor_text(img_pdf_path, 1, pdf_engine="pdfreport")
        print(image_only_anchor)

        # Parse page dimensions from both anchors and check with tolerance
        # Extract page dimensions and image bounds
        img_lines = image_only_anchor.strip().split("\n")
        len_lines = lenneg1_anchor.strip().split("\n")

        img_page_match = re.search(r"Page dimensions: ([\d.]+)x([\d.]+)", img_lines[0])
        img_image_match = re.search(r"\[Image \d+x\d+ to (\d+)x(\d+)\]", img_lines[1])

        len_page_match = re.search(r"Page dimensions: ([\d.]+)x([\d.]+)", len_lines[0])
        len_image_match = re.search(r"\[Image \d+x\d+ to (\d+)x(\d+)\]", len_lines[1])

        self.assertIsNotNone(img_page_match, f"Could not parse image anchor page dims: {image_only_anchor}")
        self.assertIsNotNone(img_image_match, f"Could not parse image anchor image dims: {image_only_anchor}")
        self.assertIsNotNone(len_page_match, f"Could not parse lenneg1 anchor page dims: {lenneg1_anchor}")
        self.assertIsNotNone(len_image_match, f"Could not parse lenneg1 anchor image dims: {lenneg1_anchor}")

        img_page_w, img_page_h = float(img_page_match.group(1)), float(img_page_match.group(2))
        img_img_w, img_img_h = int(img_image_match.group(1)), int(img_image_match.group(2))

        len_page_w, len_page_h = float(len_page_match.group(1)), float(len_page_match.group(2))
        len_img_w, len_img_h = int(len_image_match.group(1)), int(len_image_match.group(2))

        # Check page dimensions are within 1.4 tolerance
        self.assertAlmostEqual(img_page_w, len_page_w, delta=1.4, msg=f"Page width mismatch: {img_page_w} vs {len_page_w}")
        self.assertAlmostEqual(img_page_h, len_page_h, delta=1.4, msg=f"Page height mismatch: {img_page_h} vs {len_page_h}")

        # Check image dimensions are within 1 point tolerance
        self.assertAlmostEqual(img_img_w, len_img_w, delta=1, msg=f"Image width mismatch: {img_img_w} vs {len_img_w}")
        self.assertAlmostEqual(img_img_h, len_img_h, delta=1, msg=f"Image height mismatch: {img_img_h} vs {len_img_h}")

        self.assertEqual(image_only_anchor[:5], lenneg1_anchor[:5])
        self.assertEqual(image_only_anchor[-1:], lenneg1_anchor[-1:])

    def testCannotLoad(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "load_v_error.pdf")

        reader = PdfReader(local_pdf_path)
        page = 5
        anchor_text = get_anchor_text(local_pdf_path, page, pdf_engine="pdfreport", target_length=6000)

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 6000)

    @unittest.skip("TODO, this unit test still fails, the map text is too large.")
    def testExcessiveMapAnchor(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "map1.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport", target_length=6000)

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 4000)

    def testKyleOnePageAnchors1(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "dolma-page-1.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport", target_length=6000)

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 6000)

    def testKyleOnePageAnchors2(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "olmo-page-1.pdf")

        anchor_text = get_anchor_text(local_pdf_path, 1, pdf_engine="pdfreport", target_length=6000)

        print(anchor_text)
        print(len(anchor_text))
        self.assertLessEqual(len(anchor_text), 6000)


class BuildSilverTest(unittest.TestCase):
    def testSmallPage(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "small_page_size.pdf")

        from olmocr.data.buildsilver import build_page_query

        result = build_page_query(local_pdf_path, "s3://test.pdf", 1)

        from olmocr.data.renderpdf import get_png_dimensions_from_base64

        base64data = result["body"]["messages"][0]["content"][1]["image_url"]["url"]

        if base64data.startswith("data:image/png;base64,"):
            base64data = base64data[22:]

        width, height = get_png_dimensions_from_base64(base64data)

        print(width, height)

        assert max(width, height) == 2048


class TestRenderPdf(unittest.TestCase):
    def testFastMediaBoxMatchesPyPdf(self):
        for file in glob.glob(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "*.pdf")):
            reader = PdfReader(file)
            print("checking", file)

            for page_num in range(1, len(reader.pages) + 1):
                w1, h1 = get_pdf_media_box_width_height(file, page_num)
                pypdfpage = reader.pages[page_num - 1]

                self.assertAlmostEqual(w1, pypdfpage.mediabox.width, places=3)
                self.assertAlmostEqual(h1, pypdfpage.mediabox.height, places=3)


class TestOutputSamplePage(unittest.TestCase):
    def testTobaccoPaper(self):
        local_pdf_path = os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "tobacco_missed_tokens_pg1.pdf")
        anchor_text = get_anchor_text(local_pdf_path, 1, "pdfreport", target_length=6000)

        print("")
        print(anchor_text)
        print("")
