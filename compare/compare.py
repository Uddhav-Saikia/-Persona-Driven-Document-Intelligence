import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def flatten_json(obj, parent_key=''):
    items = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            # Skip any metadata keys
            if 'metadata' in new_key.split('.'):
                continue
            items.extend(flatten_json(v, new_key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}[{i}]"
            items.extend(flatten_json(v, new_key))
    else:
        items.append(f"{parent_key}: {obj}")
    return items


def get_similarity_score(json1, json2):
    flat1 = " ".join(flatten_json(json1))
    flat2 = " ".join(flatten_json(json2))

    vectorizer = TfidfVectorizer().fit([flat1, flat2])
    vecs = vectorizer.transform([flat1, flat2])
    score = cosine_similarity(vecs[0], vecs[1])[0][0]
    return round(score * 100, 2)

def compare_outputs(base_path="."):
    collections_found = False

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        if os.path.isdir(folder_path) and folder.startswith("Collection"):
            collections_found = True
            actual_file = os.path.join(folder_path, "challenge1b_output.json")
            reference_file = os.path.join(base_path, "reference", folder, "challenge1b_output.json")

            print(f"\n🔍 Checking folder: {folder}")
            print(f"   ↪ Actual: {actual_file}")
            print(f"   ↪ Reference: {reference_file}")

            if os.path.exists(actual_file) and os.path.exists(reference_file):
                try:
                    actual_json = load_json(actual_file)
                    reference_json = load_json(reference_file)
                    score = get_similarity_score(reference_json, actual_json)
                    print(f"✅ Similarity with {folder}: {score}%")
                except Exception as e:
                    print(f"⚠️ Error comparing {folder}: {e}")
            else:
                if not os.path.exists(actual_file):
                    print(f"❌ Actual file missing: {actual_file}")
                if not os.path.exists(reference_file):
                    print(f"❌ Reference file missing: {reference_file}")

    if not collections_found:
        print("🚫 No collection folders found.")

# 🔧 Run the comparison
compare_outputs()
