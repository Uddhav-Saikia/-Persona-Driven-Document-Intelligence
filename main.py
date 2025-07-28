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
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.extractor import extract_sections

warnings.filterwarnings("ignore", category=FutureWarning)

nltk.download('punkt')
nltk.download('stopwords')

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME, device="cpu")


def extract_keywords(text):
    words = nltk.word_tokenize(text.lower())
    return [w for w in words if w.isalpha() and w not in stopwords.words('english')]


def extract_keywords_from_titles(titles, top_k=15):
    titles = [t.strip() for t in titles if len(t.strip()) > 2]
    if not titles:
        return set()
    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(titles)
        scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.sum(axis=0).tolist()[0])
        sorted_keywords = sorted(scores, key=lambda x: x[1], reverse=True)
        return {kw for kw, _ in sorted_keywords[:top_k]}
    except ValueError:
        return set()


def keyword_score(text, keywords):
    lowered = text.lower()
    return sum(1 for k in keywords if k in lowered) / max(len(keywords), 1)


def semantic_score(text, job_embedding):
    return cosine_similarity(model.encode([text]), job_embedding)[0][0]


def rank_sections(sections, job_text, persona_text):
    job_keywords = extract_keywords(job_text + " " + persona_text)
    title_keywords = extract_keywords_from_titles([s["section_title"] for s in sections])
    job_embedding = model.encode([job_text])

    for s in sections:
        ks = keyword_score(s["content"], set(job_keywords) | title_keywords)
        ss = semantic_score(s["content"], job_embedding)
        s["score"] = 0.75 * ss + 0.25 * ks

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
        paragraphs = re.split(r"\n+|•|\d\.\s+|(?<=[.?!])\s{2,}", s["content"])
        clean_paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 30]
        scored_paras = []
        for p in clean_paragraphs:
            ks = keyword_score(p, job_keywords)
            ss = semantic_score(p, job_embedding)
            final = 0.75 * ss + 0.25 * ks
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
