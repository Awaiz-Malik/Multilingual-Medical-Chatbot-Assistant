import os
import fitz
import json
import time
import subprocess
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from embedding import labse_embedding_model, xlm_embedding_model, multilingual_embedding_model
from chunking import fixed_chunk, overlap_chunk, semantic_chunk_nltk, dynamic_chunk, delimiter_chunk

# Downloading dataset to directory
os.makedirs('/Users/Lenovo/Downloads/Awaiz_Temp/NLP_assignment/dataset/urdu', exist_ok=True)
os.makedirs('/Users/Lenovo/Downloads/Awaiz_Temp/NLP_assignment/dataset/spanish', exist_ok=True)

commands = [
    'curl -L -o "C:/Users/Lenovo/Downloads/Awaiz_Temp/NLP_assignment/dataset/urdu/rabies.pdf" https://www.immunize.org/wp-content/uploads/vis/urdu_rabies.pdf',
    'curl -L -o "C:/Users/Lenovo/Downloads/Awaiz_Temp/NLP_assignment/dataset/urdu/anxiety.pdf" https://www.fda.gov/media/173108/download?attachment',
    'curl -L -o "C:/Users/Lenovo/Downloads/Awaiz_Temp/NLP_assignment/dataset/spanish/xray.pdf" https://www.cda.org/wp-content/uploads/xrays_spanish.pdf',
    'curl -L -o "C:/Users/Lenovo/Downloads/Awaiz_Temp/NLP_assignment/dataset/spanish/cancer.pdf" https://www.cancer.org/content/dam/cancer-org/cancer-control/es/booklets-flyers/chemotherapy-for-cancer.pdf',
]

for cmd in commands:
    subprocess.run(cmd, shell=True, check=True)

# Cleaning and parsing each document

def doc_parse(file_path):
    pdf_document = fitz.open(file_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        txt = page.get_text()
        text += txt + "\n"
    pdf_document.close()
    return text

def doc_parse1(file_path):
    def invert_urdu_text(text):
        lines = text.split('\n')
        inverted_lines = []
        for line in lines:
            words = line.split()
            inverted_words = words[::-1]  # Reverse words in each line
            inverted_lines.append(" ".join(inverted_words))
        return "\n".join(inverted_lines)

    pdf_document = fitz.open(file_path)
    urdu_rabies = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text = page.get_text()
        inverted_txt = invert_urdu_text(text)
        urdu_rabies += inverted_txt + "\n"
    pdf_document.close()
    return urdu_rabies

def doc_parse2(file_path):
    pdf_document = fitz.open(file_path)
    spanish_xray = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        rect = page.rect
        left_rect = fitz.Rect(rect.x0, rect.y0, rect.x1 / 2, rect.y1)
        right_rect = fitz.Rect(rect.x1 / 2, rect.y0, rect.x1, rect.y1)
        left_text = page.get_text("text", clip=left_rect)
        right_text = page.get_text("text", clip=right_rect)
        spanish_xray += left_text + "\n" + right_text + "\n"
    pdf_document.close()
    return spanish_xray

spanish_cancer = doc_parse("dataset/spanish/cancer.pdf")
spanish_xray = doc_parse2("dataset/spanish/xray.pdf")
urdu_anxiety = doc_parse("dataset/urdu/anxiety.pdf")
urdu_rabies = doc_parse1("dataset/urdu/rabies.pdf")

# Chunking

documents = {
    "spanish_xray": spanish_xray,
    "spanish_cancer": spanish_cancer,
    "urdu_rabies": urdu_rabies,
    "urdu_anxiety": urdu_anxiety
}

lang_map = {
    "spanish_xray": "spanish",
    "spanish_cancer": "spanish",
    "urdu_rabies": "english", # Urdu is not available for language mapping in NLTK
    "urdu_anxiety": "english"
}

chunked_documents = {}

for doc_name, doc_text in documents.items():
    lang = lang_map.get(doc_name, "english")

    chunked_documents[doc_name] = {}
    chunked_documents[doc_name]["fixed"] = fixed_chunk(doc_text, chunk_size=100)
    chunked_documents[doc_name]["overlap"] = overlap_chunk(doc_text, chunk_size=100, overlap=20)
    chunked_documents[doc_name]["semantic_nltk"] = semantic_chunk_nltk(doc_text, lang=lang, max_words=100)
    chunked_documents[doc_name]["dynamic"] = dynamic_chunk(doc_text, lang=lang, max_tokens=200, min_tokens=50)
    chunked_documents[doc_name]["delimiter"] = delimiter_chunk(doc_text)

# Embeddings

model_labse = SentenceTransformer('sentence-transformers/LaBSE')
model_xlm = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
model_multil = SentenceTransformer('intfloat/multilingual-e5-large')

vector_store_labse = labse_embedding_model(chunked_documents, model_labse,lang_map)
vector_store_xlm = xlm_embedding_model(chunked_documents, model_xlm,lang_map)
vector_store_multili = multilingual_embedding_model(chunked_documents, model_multil,lang_map)

# Integrating pinecone

PINECONE_API_KEY = "pcsk_6rDter_H7ofP7uZadDpWiQonwHAVm76fFbmhPvrWEnqAeRZfsv6kRSnZNo5wumHRGiq5KT"
pinecone_host_link1 = "https://multilingual768-i22y2e5.svc.aped-4627-b74a.pinecone.io"
pinecone_host_link2 = "https://multilingual1024-i22y2e5.svc.aped-4627-b74a.pinecone.io"
pinecone_index_name1 = "multilingual768"
pinecone_index_name2 = "multilingual1024"

pc = Pinecone(api_key=PINECONE_API_KEY)
index1 = pc.Index(host=pinecone_host_link1)
index2 = pc.Index(host=pinecone_host_link2)

if pinecone_index_name1 and pinecone_index_name2 not in pc.list_indexes().names(): # Check if index exists
    raise Exception(f"Index may not exist.")

# Uploading vetors stores to VDB

namespaces1 = []
namespaces2 = []

for chunk_type, vectors in vector_store_labse.items():
    index1.upsert(vectors=vectors, namespace=f"{chunk_type}_labse")
    namespaces1.append(f"{chunk_type}_labse")

for chunk_type, vectors in vector_store_xlm.items():
    index1.upsert(vectors=vectors, namespace=f"{chunk_type}_xlm")
    namespaces1.append(f"{chunk_type}_xlm")
    
for chunk_type, vectors in vector_store_multili.items():
    index2.upsert(vectors=vectors, namespace=f"{chunk_type}_multilin")
    namespaces2.append(f"{chunk_type}_multilin")

# Similarity search

def clean_response(response):
    """Convert Pinecone response into JSON-serializable structure."""
    cleaned = {
        "namespace": response.get("namespace", ""),
        "matches": []
    }
    for match in response.get("matches", []):
        cleaned["matches"].append({
            "id": match["id"],
            "score": float(match["score"]),
            "metadata": match.get("metadata", {})
        })
    return cleaned

def query_index(index, namespace, model, query_text, top_k=3):
    start_time = time.time()
    query_vector = model.encode(query_text).tolist()
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    return response, elapsed_ms

spanish_query = "¿Cómo funciona la quimioterapia?"
urdu_query = "اضطراب کیا ہے؟"
results = {}

for namespace in namespaces1[:5]:   # Query for LaBSE
    res1, time1 = query_index(index1, namespace, model_labse, spanish_query)
    res2, time2 = query_index(index1, namespace, model_labse, urdu_query)
    results[namespace] = {
        spanish_query: {
            "response": clean_response(res1),
            "time_ms": round(time1, 2)
        },
        urdu_query: {
            "response": clean_response(res2),
            "time_ms": round(time2, 2)
        }
    }

for namespace in namespaces1[5:]:   # Query for XLM
    res1, time1 = query_index(index1, namespace, model_xlm, spanish_query)
    res2, time2 = query_index(index1, namespace, model_xlm, urdu_query)
    results[namespace] = {
        spanish_query: {
            "response": clean_response(res1),
            "time_ms": round(time1, 2)
        },
        urdu_query: {
            "response": clean_response(res2),
            "time_ms": round(time2, 2)
        }
    }

for namespace in namespaces2:   # Query for multilingual-e5
    res1, time1 = query_index(index2, namespace, model_multil, spanish_query)
    res2, time2 = query_index(index2, namespace, model_multil, urdu_query)
    results[namespace] = {
        spanish_query: {
            "response": clean_response(res1),
            "time_ms": round(time1, 2)
        },
        urdu_query: {
            "response": clean_response(res2),
            "time_ms": round(time2, 2)
        }
    }
    
with open("query_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)