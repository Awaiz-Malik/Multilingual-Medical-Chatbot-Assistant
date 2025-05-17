from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer models
model_labse = SentenceTransformer('sentence-transformers/LaBSE')
model_xlm = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
model_multil = SentenceTransformer('intfloat/multilingual-e5-large')


source_links = {
    "urdu_rabies": "https://www.immunize.org/wp-content/uploads/vis/urdu_rabies.pdf",
    "urdu_anxiety": "https://www.fda.gov/media/173108/download?attachment",
    "spanish_xray": "https://www.cda.org/wp-content/uploads/xrays_spanish.pdf",
    "spanish_cancer": "https://www.cancer.org/content/dam/cancer-org/cancer-control/es/booklets-flyers/chemotherapy-for-cancer.pdf"
}

embedding_model_name = {
    "labse": "LaBSE",
    "xlm": "stsb-xlm-r-multilingual",
    "multil": "multilingual-e5-large"
}

chunk_types = ["fixed", "overlap", "semantic_nltk", "dynamic", "delimiter"]

def labse_embedding_model(chunked_documents, lang_map):
    vector_stores = {chunk_type: [] for chunk_type in chunk_types}        
    for doc_name, chunks in chunked_documents.items():
        lang = lang_map.get(doc_name)
        source_link = source_links.get(doc_name, "")
        embed_model = embedding_model_name['labse']

        for chunk_type in chunk_types:
            i = 0
            for chunk in chunks[chunk_type]:
                vector_stores[chunk_type].append({
                    "id": f"{i}",
                    "values": model_labse.encode(chunk),
                    "metadata": {
                        "text": chunk,
                        "source_link": source_link,
                        "lang": lang,
                        "embedding_model_name": embed_model
                    }
                })
                i += 1
    return vector_stores
                
def xlm_embedding_model(chunked_documents, lang_map):
    vector_stores = {chunk_type: [] for chunk_type in chunk_types}        
    for doc_name, chunks in chunked_documents.items():
        lang = lang_map.get(doc_name)
        source_link = source_links.get(doc_name, "")
        embed_model = embedding_model_name['xlm']

        for chunk_type in chunk_types:
            i = 0
            for chunk in chunks[chunk_type]:
                vector_stores[chunk_type].append({
                    "id": f"{i}",
                    "values": model_xlm.encode(chunk),
                    "metadata": {
                        "text": chunk,
                        "source_link": source_link,
                        "lang": lang,
                        "embedding_model_name": embed_model
                    }
                })
                i += 1
    return vector_stores
                
def multilingual_embedding_model(chunked_documents, lang_map):
    vector_stores = {chunk_type: [] for chunk_type in chunk_types}        
    for doc_name, chunks in chunked_documents.items():
        lang = lang_map.get(doc_name)
        source_link = source_links.get(doc_name, "")
        embed_model = embedding_model_name['multil']

        for chunk_type in chunk_types:
            i = 0
            for chunk in chunks[chunk_type]:
                vector_stores[chunk_type].append({
                    "id": f"{i}",
                    "values": model_multil.encode(chunk),
                    "metadata": {
                        "text": chunk,
                        "source_link": source_link,
                        "lang": lang,
                        "embedding_model_name": embed_model
                    }
                })
                i += 1
    return vector_stores