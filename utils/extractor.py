# import os
# import fitz
# import re


# def is_heading(text, font_size, font_flags):
#     if len(text.strip()) < 3:
#         return False
#     if font_size >= 12 and (font_flags & 2):
#         return True
#     if font_size >= 14:
#         return True
#     if re.match(r'^[A-Z][A-Za-z0-9\s:,&\-]{3,}$', text) and len(text.split()) <= 10:
#         return True
#     return False

# def extract_sections(pdf_file):
#     sections = []
#     doc = fitz.open(pdf_file)
#     current_section = None

#     for page_num, page in enumerate(doc, start=1):
#         blocks = page.get_text("dict")["blocks"]

#         for b in blocks:
#             if "lines" not in b:
#                 continue
#             for line in b["lines"]:
#                 line_text = ""
#                 max_font_size = 0
#                 font_flags = 0

#                 for span in line["spans"]:
#                     line_text += span["text"].strip() + " "
#                     if span["size"] > max_font_size:
#                         max_font_size = span["size"]
#                         font_flags = span["flags"]

#                 line_text = line_text.strip()
#                 if not line_text or line_text.startswith("•"):
#                     continue

#                 if is_heading(line_text, max_font_size, font_flags):
#                     current_section = {
#                         "document": os.path.basename(pdf_file),
#                         "page_number": page_num,
#                         "section_title": line_text,
#                         "content": ""
#                     }
#                     sections.append(current_section)
#                 elif current_section:
#                     current_section["content"] += " " + line_text

#     for s in sections:
#         title = s["section_title"].strip()
#         if len(title) <= 2 or title == "•":
#             fallback = s["content"].strip().split(".")[0]
#             s["section_title"] = fallback[:80] if fallback else "Untitled Section"

#     return sections


# import os
# import fitz
# import re

# def is_heading(text, font_size, font_flags):
#     if len(text.strip()) < 3:
#         return False
#     if font_size >= 14 or (font_size >= 12 and (font_flags & 2)):
#         return True
#     if re.match(r'^[A-Z][A-Za-z0-9\s:,&\-]{3,}$', text) and len(text.split()) <= 10:
#         return True
#     return False

# def extract_sections(pdf_file):
#     doc = fitz.open(pdf_file)
#     sections = []
#     current_section = None

#     for page_num, page in enumerate(doc, start=1):
#         blocks = page.get_text("dict")["blocks"]

#         for b in blocks:
#             if "lines" not in b:
#                 continue

#             for line in b["lines"]:
#                 line_text = ""
#                 max_font_size = 0
#                 font_flags = 0

#                 for span in line["spans"]:
#                     line_text += span["text"].strip() + " "
#                     if span["size"] > max_font_size:
#                         max_font_size = span["size"]
#                         font_flags = span["flags"]

#                 line_text = line_text.strip()
#                 if not line_text or line_text.startswith("•"):
#                     continue

#                 if is_heading(line_text, max_font_size, font_flags):
#                     if current_section and not current_section["content"].strip():
#                         sections.pop()
#                     current_section = {
#                         "document": os.path.basename(pdf_file),
#                         "page_number": page_num,
#                         "section_title": line_text,
#                         "content": ""
#                     }
#                     sections.append(current_section)
#                 elif current_section:
#                     current_section["content"] += " " + line_text

#     # Clean up section titles
#     final_sections = []
#     seen = set()
#     for s in sections:
#         title = s["section_title"].strip()
#         if len(title) <= 2 or title.lower() in seen:
#             continue
#         seen.add(title.lower())
#         s["section_title"] = title
#         s["content"] = s["content"].strip()
#         final_sections.append(s)

#     return final_sections

import os
import fitz  # PyMuPDF
import re

def is_heading(text, font_size, font_flags):
    """
    Heuristic to determine if a line is a heading based on:
    - Font size
    - Font weight (bold = flags & 2)
    - Length
    - Structure
    """
    if len(text.strip()) < 3:
        return False
    if font_size >= 14:
        return True
    if font_size >= 12 and (font_flags & 2):  # bold
        return True
    if re.match(r'^[A-Z][A-Za-z0-9\s:,\-()]{3,}$', text) and len(text.split()) <= 10:
        return True
    return False

def extract_sections(pdf_file):
    sections = []
    doc = fitz.open(pdf_file)
    current_section = None

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                line_text = ""
                max_font_size = 0
                font_flags = 0

                for span in line["spans"]:
                    line_text += span["text"].strip() + " "
                    if span["size"] > max_font_size:
                        max_font_size = span["size"]
                        font_flags = span["flags"]

                line_text = line_text.strip()
                if not line_text or line_text.startswith("•"):
                    continue

                if is_heading(line_text, max_font_size, font_flags):
                    current_section = {
                        "document": os.path.basename(pdf_file),
                        "page_number": page_num,
                        "section_title": line_text,
                        "content": ""
                    }
                    sections.append(current_section)
                elif current_section:
                    current_section["content"] += " " + line_text

    # Fallback titles for empty/generic ones
    for s in sections:
        title = s["section_title"].strip()
        if len(title) <= 2 or title.lower() in {"•", "introduction", "untitled section"}:
            fallback = s["content"].strip().split(".")[0]
            s["section_title"] = fallback[:80] if fallback else "Untitled Section"

    return sections
