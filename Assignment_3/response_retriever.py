import os
import json
import numpy as np
from groq import Groq
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity

GROQ_API = "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"
CACHE_FILE = "response_cache.json"
NAMESPACE = "medical_dataset"
TOP_K = 15

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        
response_cache = load_cache()

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return None

def get_relevant_info(query: str, index, model, namespace: str = NAMESPACE, top_k: int = TOP_K):
    query_vector = model.encode(query)
    response = index.query(
        vector=query_vector.tolist(),
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )

    retrieved_chunks = []
    for match in response['matches']:
        retrieved_chunks.append({
            "text": match['metadata']['text'],
            "source_link": match['metadata'].get('source_link', 'N/A')
        })

    doc_texts = [chunk["text"] for chunk in retrieved_chunks]
    if not doc_texts:
        return []

    doc_vectors = model.encode(doc_texts)

    # Cosine similarity and top 3 selection
    sims = cosine_similarity([query_vector], doc_vectors).flatten()
    top_indices = np.argsort(sims)[-3:][::-1]

    top_chunks = [retrieved_chunks[i] for i in top_indices]
    return top_chunks

def medical_expert_response(query: str, index, model) -> str:
    query_key = query.strip()
    if query_key in response_cache:
        print("[CACHE HIT]")
        return response_cache[query_key]

    print("[CACHE MISS]")

    client = Groq(
        api_key=GROQ_API,
    )

    detected_lang = detect_language(query)

    if not detected_lang:
        return "I am unable to detect the language of your query. Please try again with proper prompt."

    system_prompt = f"""You are a highly knowledgeable medical expert.
                      Answer the user's query concisely and directly.
                      Ensure the response is in {detected_lang} language as the user's query."""

    relevant_info = get_relevant_info(query, index, model)

    if not relevant_info:
        return "I am unable to find relevant information for your query at this time. Please consult a medical professional for advice."

    # Concatenate relevant text for a more comprehensive context for the LLM
    context = " ".join([info['text'] for info in relevant_info])
    
    chat_completion = client.chat.completions.create(
    messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"User Query: {query}\nRelevant Information: {context}"
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )

    final_response = chat_completion.choices[0].message.content
    response_cache[query_key] = final_response
    save_cache(response_cache)
    
    return final_response
