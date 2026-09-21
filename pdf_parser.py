import pymupdf
from openai import OpenAI

class PDFExtractor():
    def pdf_text_extractor(pdf_path):
        page_text = []
        doc = pymupdf.open(pdf_path)

        for page in doc:
            text = page.get_text()
            page_text.append(text)
        
        return page_text
