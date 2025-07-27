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



# import os
# import json
# import fitz
# import re
# import nltk
# import warnings
# from datetime import datetime
# from nltk.corpus import stopwords
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from collections import defaultdict

# warnings.filterwarnings("ignore", category=FutureWarning)

# # ---------------- NLTK Downloads ----------------
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('stopwords')

# # ---------------- MODEL ----------------
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# model = SentenceTransformer(MODEL_NAME, device="cpu")

# # ---------------- PDF SECTION EXTRACTION ----------------
# def is_probable_heading(block):
#     """
#     Determines if a block is likely a heading based on:
#     - Noun/adjective ratio (dish names & section titles are noun-heavy)
#     - Short length (<=8 words)
#     - No trailing punctuation
#     """
#     words = nltk.word_tokenize(block)
#     if len(words) == 0 or len(words) > 8 or block.endswith("."):
#         return False
#     pos_tags = nltk.pos_tag(words)
#     noun_like = sum(1 for _, pos in pos_tags if pos.startswith(("NN", "JJ")))
#     verb_like = sum(1 for _, pos in pos_tags if pos.startswith("VB"))
#     return noun_like >= verb_like

# def extract_sections(pdf_file):
#     sections = []
#     doc = fitz.open(pdf_file)

#     for page_num, page in enumerate(doc, start=1):
#         text = page.get_text()
#         current_section = None

#         for block in text.split("\n"):
#             block = block.strip()
#             if not block:
#                 continue

#             if is_probable_heading(block):
#                 current_section = {
#                     "document": os.path.basename(pdf_file),
#                     "page_number": page_num,
#                     "section_title": block,
#                     "content": ""
#                 }
#                 sections.append(current_section)
#             elif current_section:
#                 current_section["content"] += " " + block

#     # Fallback for bullet headings or empty titles
#     for s in sections:
#         if s["section_title"].strip() in ["•", ""]:
#             first_sentence = s["content"].strip().split(".")[0]
#             s["section_title"] = first_sentence[:80] if first_sentence else "Untitled Section"
#     return sections

# # ---------------- NLP HELPERS ----------------
# def extract_keywords(text):
#     words = nltk.word_tokenize(text.lower())
#     return [w for w in words if w.isalpha() and w not in stopwords.words('english')]

# def keyword_score(text, keywords):
#     return sum(1 for k in keywords if k in text.lower()) / max(len(keywords), 1)

# def semantic_score(text, job_embedding):
#     return cosine_similarity(model.encode([text]), job_embedding)[0][0]

# def dynamic_diet_penalty(text, job_text):
#     negatives = []
#     jt = job_text.lower()
#     if "vegetarian" in jt:
#         negatives += ["beef", "chicken", "pork", "meat", "bacon"]
#     if "gluten" in jt:
#         negatives += ["bread", "wheat", "pasta", "flour"]
#     return -0.3 if any(n in text.lower() for n in negatives) else 0

# # ---------------- RANKING ----------------
# def rank_sections(sections, job_text, persona_text):
#     job_keywords = extract_keywords(job_text + " " + persona_text)
#     job_embedding = model.encode([job_text])

#     for s in sections:
#         title_score = semantic_score(s["section_title"], job_embedding)
#         content_score = semantic_score(s["content"], job_embedding)
#         ks = keyword_score(s["content"], job_keywords)
#         penalty = dynamic_diet_penalty(s["content"], job_text)

#         s["score"] = 0.5 * title_score + 0.3 * content_score + 0.2 * ks + penalty

#     ranked = sorted(sections, key=lambda x: x["score"], reverse=True)
#     for i, s in enumerate(ranked, start=1):
#         s["importance_rank"] = i
#     return ranked

# def select_diverse_sections(ranked, top_per_doc=1, overall_top=5):
#     selected = []
#     seen_docs = defaultdict(int)
#     for s in ranked:
#         if seen_docs[s["document"]] < top_per_doc:
#             selected.append(s)
#             seen_docs[s["document"]] += 1
#         if len(selected) >= overall_top:
#             break
#     return selected

# # ---------------- SUBSECTION EXTRACTION ----------------
# def extract_subsections(top_sections, job_text, top_n=3):
#     job_keywords = extract_keywords(job_text)
#     job_embedding = model.encode([job_text])
#     sub_analysis = []
#     for s in top_sections:
#         paragraphs = [p.strip() for p in s["content"].split(". ") if len(p.strip()) > 20]
#         scored_paras = []
#         for p in paragraphs:
#             ks = keyword_score(p, job_keywords)
#             ss = semantic_score(p, job_embedding)
#             final = 0.6 * ss + 0.4 * ks
#             scored_paras.append((final, p))
#         scored_paras = sorted(scored_paras, key=lambda x: x[0], reverse=True)[:top_n]
#         for _, para in scored_paras:
#             sub_analysis.append({
#                 "document": s["document"],
#                 "page_number": s["page_number"],
#                 "refined_text": para
#             })
#     return sub_analysis

# # ---------------- MAIN PROCESS ----------------
# def process_collection(collection_path):
#     input_path = os.path.join(collection_path, "challenge1b_input.json")
#     output_path = os.path.join(collection_path, "challenge1b_output.json")

#     with open(input_path, "r", encoding="utf-8") as f:
#         config = json.load(f)

#     persona = config["persona"]["role"]
#     job_text = config["job_to_be_done"]["task"]

#     pdf_dir = os.path.join(collection_path, "pdfs")
#     pdf_files = [os.path.join(pdf_dir, doc["filename"]) for doc in config["documents"]]

#     all_sections = []
#     for pdf in pdf_files:
#         all_sections.extend(extract_sections(pdf))

#     ranked = rank_sections(all_sections, job_text, persona)
#     top_sections = select_diverse_sections(ranked)

#     extracted_sections = [{
#         "document": s["document"],
#         "section_title": s["section_title"],
#         "importance_rank": s["importance_rank"],
#         "page_number": s["page_number"]
#     } for s in top_sections]

#     sub_section_analysis = extract_subsections(top_sections, job_text)

#     output = {
#         "metadata": {
#             "input_documents": [doc["filename"] for doc in config["documents"]],
#             "persona": persona,
#             "job_to_be_done": job_text,
#             "processing_timestamp": datetime.now().isoformat()
#         },
#         "extracted_sections": extracted_sections,
#         "subsection_analysis": sub_section_analysis
#     }

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(output, f, indent=4, ensure_ascii=False)

#     print(f"✅ Output saved to {output_path}")

# if __name__ == "__main__":
#     base_dir = os.getcwd()
#     for folder in os.listdir(base_dir):
#         if folder.lower().startswith("collection"):
#             collection_path = os.path.join(base_dir, folder)
#             input_file = os.path.join(collection_path, "challenge1b_input.json")
#             if os.path.exists(input_file):
#                 print(f"🚀 Processing {folder} ...")
#                 process_collection(collection_path)
#             else:
#                 print(f"⚠️ Skipping {folder} (no challenge1b_input.json found)")
#     print("\n✅ All collections processed!")


# import os
# import json
# import fitz
# import re
# import nltk
# import warnings
# from datetime import datetime
# from nltk.corpus import stopwords
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from collections import defaultdict, Counter

# warnings.filterwarnings("ignore", category=FutureWarning)

# # ---------------- NLTK Downloads ----------------
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("stopwords")

# # ---------------- MODEL ----------------
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# model = SentenceTransformer(MODEL_NAME, device="cpu")

# # ---------------- HEADING DETECTION ----------------
# def is_probable_heading(block, food_keywords):
#     """
#     Determines if a block is likely a heading:
#     ✅ Short length (<=8 words), no trailing punctuation
#     ✅ Noun/adjective heavy (dish names tend to be noun-based)
#     ✅ Contains food-related words if available
#     """
#     words = nltk.word_tokenize(block)
#     if len(words) == 0 or len(words) > 8 or block.endswith("."):
#         return False

#     pos_tags = nltk.pos_tag(words)
#     noun_like = sum(1 for _, pos in pos_tags if pos.startswith(("NN", "JJ")))
#     verb_like = sum(1 for _, pos in pos_tags if pos.startswith("VB"))

#     # Boost if food keywords detected
#     food_hit = any(w.lower() in food_keywords for w in words)

#     return (noun_like >= verb_like and (food_hit or noun_like >= 2))

# # ---------------- FOOD KEYWORDS EXTRACTION ----------------
# def extract_food_keywords_from_pdfs(pdf_files, top_n=50):
#     """
#     Scans all PDFs to dynamically build a list of frequent food-related nouns.
#     """
#     all_text = ""
#     for pdf in pdf_files:
#         doc = fitz.open(pdf)
#         for page in doc:
#             all_text += " " + page.get_text()

#     words = nltk.word_tokenize(all_text.lower())
#     words = [w for w in words if w.isalpha() and w not in stopwords.words("english")]
#     pos_tags = nltk.pos_tag(words)

#     food_nouns = [w for w, pos in pos_tags if pos.startswith("NN")]
#     most_common = [w for w, _ in Counter(food_nouns).most_common(top_n)]
#     return set(most_common)

# # ---------------- PDF SECTION EXTRACTION ----------------
# def extract_sections(pdf_file, food_keywords):
#     sections = []
#     doc = fitz.open(pdf_file)

#     for page_num, page in enumerate(doc, start=1):
#         text = page.get_text()
#         current_section = None

#         for block in text.split("\n"):
#             block = block.strip()
#             if not block:
#                 continue

#             if is_probable_heading(block, food_keywords):
#                 current_section = {
#                     "document": os.path.basename(pdf_file),
#                     "page_number": page_num,
#                     "section_title": block,
#                     "content": ""
#                 }
#                 sections.append(current_section)
#             elif current_section:
#                 current_section["content"] += " " + block

#     # Fallback for empty/bullet headings
#     for s in sections:
#         if s["section_title"].strip() in ["•", ""]:
#             first_sentence = s["content"].strip().split(".")[0]
#             s["section_title"] = first_sentence[:80] if first_sentence else "Untitled Section"
#     return sections

# # ---------------- NLP HELPERS ----------------
# def extract_keywords(text):
#     words = nltk.word_tokenize(text.lower())
#     return [w for w in words if w.isalpha() and w not in stopwords.words("english")]

# def keyword_score(text, keywords):
#     return sum(1 for k in keywords if k in text.lower()) / max(len(keywords), 1)

# def semantic_score(text, job_embedding):
#     return cosine_similarity(model.encode([text]), job_embedding)[0][0]

# def dynamic_diet_penalty(text, job_text):
#     negatives = []
#     jt = job_text.lower()
#     if "vegetarian" in jt:
#         negatives += ["beef", "chicken", "pork", "meat", "bacon", "fish"]
#     if "gluten" in jt:
#         negatives += ["bread", "wheat", "pasta", "flour", "tortilla"]
#     return -0.4 if any(n in text.lower() for n in negatives) else 0

# # ---------------- RANKING ----------------
# def rank_sections(sections, job_text, persona_text):
#     job_keywords = extract_keywords(job_text + " " + persona_text)
#     job_embedding = model.encode([job_text])

#     for s in sections:
#         title_score = semantic_score(s["section_title"], job_embedding)
#         content_score = semantic_score(s["content"], job_embedding)
#         ks = keyword_score(s["content"], job_keywords)
#         penalty = dynamic_diet_penalty(s["content"], job_text)

#         # Titles (dish names) get higher weight
#         s["score"] = 0.6 * title_score + 0.25 * content_score + 0.15 * ks + penalty

#     ranked = sorted(sections, key=lambda x: x["score"], reverse=True)
#     for i, s in enumerate(ranked, start=1):
#         s["importance_rank"] = i
#     return ranked

# def select_diverse_sections(ranked, top_per_doc=1, overall_top=5):
#     selected = []
#     seen_docs = defaultdict(int)
#     seen_titles = set()
#     for s in ranked:
#         title_key = s["section_title"].lower()
#         if seen_docs[s["document"]] < top_per_doc and title_key not in seen_titles:
#             selected.append(s)
#             seen_docs[s["document"]] += 1
#             seen_titles.add(title_key)
#         if len(selected) >= overall_top:
#             break
#     return selected

# # ---------------- SUBSECTION EXTRACTION ----------------
# def extract_subsections(top_sections, job_text, top_n=3):
#     job_keywords = extract_keywords(job_text)
#     job_embedding = model.encode([job_text])
#     sub_analysis = []
#     for s in top_sections:
#         paragraphs = [p.strip() for p in s["content"].split(". ") if len(p.strip()) > 20]
#         scored_paras = []
#         for p in paragraphs:
#             ks = keyword_score(p, job_keywords)
#             ss = semantic_score(p, job_embedding)
#             final = 0.6 * ss + 0.4 * ks
#             scored_paras.append((final, p))
#         scored_paras = sorted(scored_paras, key=lambda x: x[0], reverse=True)[:top_n]
#         for _, para in scored_paras:
#             sub_analysis.append({
#                 "document": s["document"],
#                 "page_number": s["page_number"],
#                 "refined_text": para
#             })
#     return sub_analysis

# # ---------------- MAIN PROCESS ----------------
# def process_collection(collection_path):
#     input_path = os.path.join(collection_path, "challenge1b_input.json")
#     output_path = os.path.join(collection_path, "challenge1b_output.json")

#     with open(input_path, "r", encoding="utf-8") as f:
#         config = json.load(f)

#     persona = config["persona"]["role"]
#     job_text = config["job_to_be_done"]["task"]

#     pdf_dir = os.path.join(collection_path, "pdfs")
#     pdf_files = [os.path.join(pdf_dir, doc["filename"]) for doc in config["documents"]]

#     # ✅ Extract food keywords dynamically
#     food_keywords = extract_food_keywords_from_pdfs(pdf_files)

#     all_sections = []
#     for pdf in pdf_files:
#         all_sections.extend(extract_sections(pdf, food_keywords))

#     ranked = rank_sections(all_sections, job_text, persona)
#     top_sections = select_diverse_sections(ranked)

#     extracted_sections = [{
#         "document": s["document"],
#         "section_title": s["section_title"],
#         "importance_rank": s["importance_rank"],
#         "page_number": s["page_number"]
#     } for s in top_sections]

#     sub_section_analysis = extract_subsections(top_sections, job_text)

#     output = {
#         "metadata": {
#             "input_documents": [doc["filename"] for doc in config["documents"]],
#             "persona": persona,
#             "job_to_be_done": job_text,
#             "processing_timestamp": datetime.now().isoformat()
#         },
#         "extracted_sections": extracted_sections,
#         "subsection_analysis": sub_section_analysis
#     }

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(output, f, indent=4, ensure_ascii=False)

#     print(f"✅ Output saved to {output_path}")

# if __name__ == "__main__":
#     base_dir = os.getcwd()
#     for folder in os.listdir(base_dir):
#         if folder.lower().startswith("collection"):
#             collection_path = os.path.join(base_dir, folder)
#             input_file = os.path.join(collection_path, "challenge1b_input.json")
#             if os.path.exists(input_file):
#                 print(f"🚀 Processing {folder} ...")
#                 process_collection(collection_path)
#             else:
#                 print(f"⚠️ Skipping {folder} (no challenge1b_input.json found)")
#     print("\n✅ All collections processed!")


# import os
# import json
# import fitz
# import re
# import nltk
# import warnings
# from datetime import datetime
# from nltk.corpus import stopwords
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from collections import defaultdict

# warnings.filterwarnings("ignore", category=FutureWarning)

# # ---------------- NLTK Downloads ----------------
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

# # ---------------- MODEL ----------------
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# model = SentenceTransformer(MODEL_NAME, device="cpu")

# # ---------------- PDF SECTION EXTRACTION ----------------
# def is_probable_heading(block):
#     """Improved heading detection using POS tags + simple heuristics"""
#     words = nltk.word_tokenize(block)
#     if len(words) == 0 or len(words) > 8 or block.endswith("."):
#         return False
#     pos_tags = nltk.pos_tag(words)
#     noun_like = sum(1 for _, pos in pos_tags if pos.startswith(("NN", "JJ")))
#     verb_like = sum(1 for _, pos in pos_tags if pos.startswith("VB"))
#     return noun_like >= verb_like

# def extract_sections(pdf_file):
#     sections = []
#     doc = fitz.open(pdf_file)
#     for page_num, page in enumerate(doc, start=1):
#         text = page.get_text()
#         current_section = None
#         for block in text.split("\n"):
#             block = block.strip()
#             if not block:
#                 continue
#             if is_probable_heading(block):
#                 current_section = {
#                     "document": os.path.basename(pdf_file),
#                     "page_number": page_num,
#                     "section_title": block,
#                     "content": ""
#                 }
#                 sections.append(current_section)
#             elif current_section:
#                 current_section["content"] += " " + block

#     for s in sections:
#         if s["section_title"].strip() in ["•", ""]:
#             first_sentence = s["content"].strip().split(".")[0]
#             s["section_title"] = first_sentence[:80] if first_sentence else "Untitled Section"
#     return sections

# # ---------------- NLP HELPERS ----------------
# def extract_keywords(text):
#     words = nltk.word_tokenize(text.lower())
#     return [w for w in words if w.isalpha() and w not in stopwords.words('english')]

# def keyword_score(text, keywords):
#     return sum(1 for k in keywords if k in text.lower()) / max(len(keywords), 1)

# def semantic_score(text, job_embedding):
#     return cosine_similarity(model.encode([text]), job_embedding)[0][0]

# # ---------------- DIET & FOOD LOGIC ----------------
# VEG_BOOST_WORDS = ["falafel", "ratatouille", "baba", "ganoush", "sushi", "rolls",
#                    "lasagna", "vegetable", "salad", "tofu", "escalivada"]
# GLUTEN_FREE_WORDS = ["gluten-free", "quinoa", "rice", "buckwheat", "corn"]

# NON_VEG_WORDS = ["beef", "chicken", "pork", "turkey", "lamb", "meat", "bacon"]
# GLUTEN_WORDS = ["wheat", "bread", "pasta", "flour", "tortilla"]

# def diet_food_score(title, job_text):
#     title_lower = title.lower()
#     jt = job_text.lower()

#     # ✅ Strong penalties for non-veg/gluten dishes
#     if any(w in title_lower for w in NON_VEG_WORDS) and "vegetarian" in jt:
#         return -2.0
#     if any(w in title_lower for w in GLUTEN_WORDS) and "gluten" in jt:
#         return -1.5

#     # ✅ Boost for vegetarian/gluten-friendly titles
#     boost = 0.0
#     if any(w in title_lower for w in VEG_BOOST_WORDS):
#         boost += 1.0
#     if any(w in title_lower for w in GLUTEN_FREE_WORDS):
#         boost += 0.5
#     return boost

# # ---------------- RANKING ----------------
# def rank_sections(sections, job_text, persona_text):
#     job_keywords = extract_keywords(job_text + " " + persona_text)
#     job_embedding = model.encode([job_text])

#     for s in sections:
#         title_score = semantic_score(s["section_title"], job_embedding)
#         content_score = semantic_score(s["content"], job_embedding)
#         ks = keyword_score(s["content"], job_keywords)
#         diet_score = diet_food_score(s["section_title"], job_text)
#         s["score"] = 0.5 * title_score + 0.3 * content_score + 0.2 * ks + diet_score

#     ranked = sorted(sections, key=lambda x: x["score"], reverse=True)
#     for i, s in enumerate(ranked, start=1):
#         s["importance_rank"] = i
#     return ranked

# def select_diverse_sections(ranked, top_per_doc=1, overall_top=5):
#     selected = []
#     seen_docs = defaultdict(int)
#     for s in ranked:
#         if seen_docs[s["document"]] < top_per_doc:
#             selected.append(s)
#             seen_docs[s["document"]] += 1
#         if len(selected) >= overall_top:
#             break
#     return selected

# # ---------------- SUBSECTION EXTRACTION ----------------
# def extract_subsections(top_sections, job_text, top_n=3):
#     job_keywords = extract_keywords(job_text)
#     job_embedding = model.encode([job_text])
#     sub_analysis = []
#     for s in top_sections:
#         paragraphs = [p.strip() for p in s["content"].split(". ") if len(p.strip()) > 20]
#         scored_paras = []
#         for p in paragraphs:
#             ks = keyword_score(p, job_keywords)
#             ss = semantic_score(p, job_embedding)
#             final = 0.6 * ss + 0.4 * ks
#             scored_paras.append((final, p))
#         scored_paras = sorted(scored_paras, key=lambda x: x[0], reverse=True)[:top_n]
#         for _, para in scored_paras:
#             sub_analysis.append({
#                 "document": s["document"],
#                 "page_number": s["page_number"],
#                 "refined_text": para
#             })
#     return sub_analysis

# # ---------------- MAIN PROCESS ----------------
# def process_collection(collection_path):
#     input_path = os.path.join(collection_path, "challenge1b_input.json")
#     output_path = os.path.join(collection_path, "challenge1b_output.json")

#     with open(input_path, "r", encoding="utf-8") as f:
#         config = json.load(f)

#     persona = config["persona"]["role"]
#     job_text = config["job_to_be_done"]["task"]

#     pdf_dir = os.path.join(collection_path, "pdfs")
#     pdf_files = [os.path.join(pdf_dir, doc["filename"]) for doc in config["documents"]]

#     all_sections = []
#     for pdf in pdf_files:
#         all_sections.extend(extract_sections(pdf))

#     ranked = rank_sections(all_sections, job_text, persona)
#     top_sections = select_diverse_sections(ranked)

#     extracted_sections = [{
#         "document": s["document"],
#         "section_title": s["section_title"],
#         "importance_rank": s["importance_rank"],
#         "page_number": s["page_number"]
#     } for s in top_sections]

#     sub_section_analysis = extract_subsections(top_sections, job_text)

#     output = {
#         "metadata": {
#             "input_documents": [doc["filename"] for doc in config["documents"]],
#             "persona": persona,
#             "job_to_be_done": job_text,
#             "processing_timestamp": datetime.now().isoformat()
#         },
#         "extracted_sections": extracted_sections,
#         "subsection_analysis": sub_section_analysis
#     }

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(output, f, indent=4, ensure_ascii=False)

#     print(f"✅ Output saved to {output_path}")

# if __name__ == "__main__":
#     base_dir = os.getcwd()
#     for folder in os.listdir(base_dir):
#         if folder.lower().startswith("collection"):
#             collection_path = os.path.join(base_dir, folder)
#             input_file = os.path.join(collection_path, "challenge1b_input.json")
#             if os.path.exists(input_file):
#                 print(f"🚀 Processing {folder} ...")
#                 process_collection(collection_path)
#             else:
#                 print(f"⚠️ Skipping {folder} (no challenge1b_input.json found)")
#     print("\n✅ All collections processed!")
