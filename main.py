import os
import json
import fitz
import re
import nltk
import warnings
from datetime import datetime
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


warnings.filterwarnings("ignore", category=FutureWarning)

nltk.download('punkt')
nltk.download('punkt_tab')  
nltk.download('stopwords')

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


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

            if (
                not block.startswith("•") and
                (re.match(r"^[A-Z][A-Za-z0-9\s\-\&]{3,}$", block) or (3 <= len(block.split()) <= 8))
            ):
                current_section = {
                    "document": os.path.basename(pdf_file),
                    "page_number": page_num,
                    "section_title": block,
                    "content": ""
                }
                sections.append(current_section)
            elif current_section:
                current_section["content"] += " " + block

    for s in sections:
        if s["section_title"].strip() == "•" or len(s["section_title"].strip()) <= 2:
            first_sentence = s["content"].strip().split(".")[0]
            if first_sentence:
                s["section_title"] = first_sentence[:80]  
            else:
                s["section_title"] = "Untitled Section"

    return sections

def extract_keywords(text):
    words = nltk.word_tokenize(text.lower())
    return [w for w in words if w.isalpha() and w not in stopwords.words('english')]

def keyword_score(text, keywords):
    return sum(1 for k in keywords if k in text.lower()) / max(len(keywords), 1)

def semantic_score(text, job_embedding):
    return cosine_similarity(model.encode([text]), job_embedding)[0][0]

def rank_sections(sections, job_text, persona_text):
    job_keywords = extract_keywords(job_text + " " + persona_text)
    job_embedding = model.encode([job_text])

    for s in sections:
        ks = keyword_score(s["content"], job_keywords)
        ss = semantic_score(s["content"], job_embedding)
        s["score"] = 0.8 * ss + 0.2 * ks  

        if s["section_title"].strip() in ["•", "Untitled Section"]:
            s["score"] -= 0.2
        if any(word in s["document"].lower() for word in ["cities", "things", "restaurants", "cuisine"]):
            s["score"] += 0.1

    ranked = sorted(sections, key=lambda x: x["score"], reverse=True)
    for i, s in enumerate(ranked, start=1):
        s["importance_rank"] = i
    return ranked

def select_diverse_sections(ranked, top_per_doc=1, overall_top=5):
    selected = []
    seen_docs = defaultdict(int)
    for s in ranked:
        if seen_docs[s["document"]] < top_per_doc:
            selected.append(s)
            seen_docs[s["document"]] += 1
        if len(selected) >= overall_top:
            break
    return selected

def extract_subsections(top_sections, job_text, top_n=3):
    job_keywords = extract_keywords(job_text)
    job_embedding = model.encode([job_text])
    sub_analysis = []
    for s in top_sections:
        paragraphs = [p.strip() for p in s["content"].split(". ") if len(p.strip()) > 20]
        scored_paras = []
        for p in paragraphs:
            ks = keyword_score(p, job_keywords)
            ss = semantic_score(p, job_embedding)
            final = 0.8 * ss + 0.2 * ks
            scored_paras.append((final, p))
        scored_paras = sorted(scored_paras, key=lambda x: x[0], reverse=True)[:top_n]
        for _, para in scored_paras:
            sub_analysis.append({
                "document": s["document"],
                "page_number": s["page_number"],
                "refined_text": para
            })
    return sub_analysis

def process_collection(collection_path):
    input_path = os.path.join(collection_path, "challenge1b_input.json")
    output_path = os.path.join(collection_path, "challenge1b_output.json")

    with open(input_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    persona = config["persona"]["role"]
    job_text = config["job_to_be_done"]["task"]

    pdf_dir = os.path.join(collection_path, "pdfs")
    pdf_files = [os.path.join(pdf_dir, doc["filename"]) for doc in config["documents"]]

    all_sections = []
    for pdf in pdf_files:
        all_sections.extend(extract_sections(pdf))

    ranked = rank_sections(all_sections, job_text, persona)
    top_sections = select_diverse_sections(ranked)

    extracted_sections = [{
        "document": s["document"],
        "section_title": s["section_title"],
        "importance_rank": s["importance_rank"],
        "page_number": s["page_number"]
    } for s in top_sections]

    sub_section_analysis = extract_subsections(top_sections, job_text)

    output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in config["documents"]],
            "persona": persona,
            "job_to_be_done": job_text,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": sub_section_analysis
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"✅ Output saved to {output_path}")

if __name__ == "__main__":
    base_dir = os.getcwd()
    for folder in os.listdir(base_dir):
        if folder.lower().startswith("collection"):
            collection_path = os.path.join(base_dir, folder)
            input_file = os.path.join(collection_path, "challenge1b_input.json")
            if os.path.exists(input_file):
                print(f"🚀 Processing {folder} ...")
                process_collection(collection_path)
            else:
                print(f"⚠️ Skipping {folder} (no challenge1b_input.json found)")
    print("\n✅ All collections processed!")



