import os
import re
import fitz

def extract_sections(pdf_file):
    sections = []
    doc = fitz.open(pdf_file)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        current_section = None
        for block in text.split("\n"):
            block = block.strip()
            if not block:
                continue
            # Heuristic to detect headings
            if re.match(r"^[A-Z][A-Za-z0-9\s\-\&]{3,}$", block) or len(block.split()) <= 8:
                current_section = {
                    "document": os.path.basename(pdf_file),
                    "page_number": page_num,
                    "section_title": block,
                    "content": ""
                }
                sections.append(current_section)
            elif current_section:
                current_section["content"] += " " + block
    return sections