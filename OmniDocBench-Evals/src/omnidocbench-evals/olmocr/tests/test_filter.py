import os
import unittest

from pypdf import PdfReader

from olmocr.filter import PdfFilter


class PdfFilterTest(unittest.TestCase):
    def testFormLaterPages(self):
        self.filter = PdfFilter(apply_form_check=True)

        self.assertTrue(self.filter.filter_out_pdf(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "form_on_later_pages.pdf")))

        self.filter = PdfFilter(apply_form_check=False)

        self.assertFalse(self.filter.filter_out_pdf(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "form_on_later_pages.pdf")))
