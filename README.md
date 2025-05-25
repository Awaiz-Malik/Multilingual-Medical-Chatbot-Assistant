# Multilingual-Medical-Chatbot-Assistant

---

A retrieval-augmented generation (RAG) system designed to answer medical questions in **Urdu** and **Spanish** using multilingual PDF documents as its knowledge base.

---

## 📁 Project Overview

This project builds a multilingual medical chatbot that:
- Parses real-world Spanish and Urdu medical documents.
- Chunks documents using structure-aware strategies.
- Generates dense embeddings using a multilingual model.
- Stores vectorized chunks in a Pinecone vector database.
- Retrieves top-matching chunks based on cosine similarity.
- Passes them to an LLM for context-aware response generation.

---

## 📄 Data Sources

PDFs are downloaded from trusted health organizations such as:
- [FDA](https://www.fda.gov/)
- [CDC](https://www.cdc.gov/)
- [Immunize.org](https://www.immunize.org/)
- [CDA](https://www.cda.org/)
- [Reproductive Health Access](https://www.reproductiveaccess.org/)

Documents include topics on:
- Anxiety, Rabies, Asthma, Heart Disease (Urdu)
- IUDs, Infections, X-rays, Bad Breath, Blood Clots (Spanish)

---

## 🧩 Parsing Techniques

- **Standard Extraction**: Used for English/Spanish documents with linear reading order.
- **Urdu Handling**: Adjusted for right-to-left script and word order.
- **Two-Column Layout**: Spanish documents with columns are processed separately.

---

## ✂️ Chunking Strategies

| Function | Description |
|---------|-------------|
| `chunk_questions_answers()` | Extracts Q&A pairs from Spanish health forms using `¿...?` patterns. |
| `chunk_numbered_sections()` | Identifies structured sections using numeric headings in Spanish medical documents. |
| `chunk_by_number()` | Parses Urdu documents with formal enumeration (e.g., `1.`, `.2`). |
| `chunk_urdu()` | Specialized for Urdu Q&A structure using `؟` marks. |
| `character_text_splitter()` | Generic fallback for unstructured documents with customizable chunk size and overlap. |

*Chunk tuning was done using **ChunkViz** for optimal semantic segmentation.*

---

## 🔍 Embeddings & Vector Search

- Uses `intfloat/multilingual-e5-large` for cross-language dense embeddings.
- Embeddings are upserted into a **Pinecone** index with metadata (text + source link).
- Retrieval is based on cosine similarity of encoded queries and document vectors.

---

## 💬 Response Generation

- Language of user query is auto-detected.
- Top 3 most relevant chunks are selected using similarity ranking.
- Responses are generated using `meta-llama/llama-4-scout-17b-16e-instruct` via Groq API.
- All responses match the language of the user query.

---

## 🌐 Supported Languages

- ✅ Spanish (es)
- ✅ Urdu (ur)

More languages can be added with additional documents and parser logic.

---
