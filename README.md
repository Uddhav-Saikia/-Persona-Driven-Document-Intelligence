# 📄 PDF Section Extractor

This project processes PDF documents for section extraction and semantic summarization based on a given persona and task (job to be done). The system uses a combination of structural heuristics, semantic similarity (Sentence-BERT), and keyword relevance (TF-IDF) to extract meaningful content sections from documents.

---

## 🧠 Approach

1. **PDF Parsing**  
   Uses PyMuPDF to extract text spans, font sizes, and positional metadata.

2. **Heading Detection**  
   Heuristically determines headings based on:
   - Font size & boldness
   - Capitalization and structure
   - Length of text

3. **Section Extraction**  
   Text is grouped under detected headings to form meaningful sections.

4. **Keyword & Semantic Scoring**  
   - `TF-IDF` is applied to headings and content for keyword relevance.
   - `SentenceTransformers` (MiniLM-L6-v2) generates embeddings for semantic similarity.
   - Sections are ranked using a hybrid score:  
     `final_score = 0.75 * semantic_similarity + 0.25 * keyword_score`

5. **Diversity Filtering**  
   Selects top-ranked sections across documents while maintaining variety.

6. **Subsection Analysis**  
   Extracts top-ranked paragraphs from selected sections to form fine-grained analysis.

---

## 🔍 Models & Libraries Used

| Task                     | Tool / Library                       |
|--------------------------|--------------------------------------|
| PDF Text Extraction      | [`PyMuPDF`](https://pymupdf.readthedocs.io/) (`fitz`) |
| Tokenization & Stopwords| `nltk`                                |
| Semantic Embedding       | [`SentenceTransformers`](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| Keyword Scoring          | `scikit-learn` TF-IDF                |
| Similarity Scoring       | `cosine_similarity` from `scikit-learn` |

---

## ⚙️ Expected Execution (Docker)

Your container is expected to automatically process all PDFs from `/app/input`, producing a `.json` output for each `.pdf` in `/app/output`.

### 🛠️ Build the Docker image:
```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```
🚀 Run the Docker container:
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  mysolutionname:somerandomidentifier
```

### 📂 Input / Output Expectations
🔸 Input: /app/input/
```Accepts one or more .pdf files.```

🔹 Output: /app/output/
```For each filename.pdf, a corresponding filename.json is created.```

✅ Output JSON structure:
```json
{
  "metadata": {
    "input_documents": ["example.pdf"],
    "persona": "Travel Blogger",
    "job_to_be_done": "Find family-friendly weekend ideas",
    "processing_timestamp": "2025-07-28T14:23:45"
  },
  "extracted_sections": [
    {
      "document": "example.pdf",
      "section_title": "Weekend Getaways",
      "importance_rank": 1,
      "page_number": 3
    }
  ],
  "subsection_analysis": [
    {
      "document": "example.pdf",
      "page_number": 3,
      "refined_text": "Explore charming hill stations within 3 hours of the city..."
    }
  ]
}
```

📄 License
This project is released under the MIT License.
