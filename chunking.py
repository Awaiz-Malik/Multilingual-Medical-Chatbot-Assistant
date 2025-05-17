import nltk
import tiktoken
from typing import List

def count_tokens(text, model_name="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))

# 1. Fixed-size chunking
def fixed_chunk(text: str, chunk_size: int) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# 2. Overlap-based chunking
def overlap_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# 3. Semantic chunking using NLTK (sentence-based)
def semantic_chunk_nltk(text: str, lang: str, max_words: int = 100) -> List[str]:
    nltk.download('punkt_tab')
    sentences = nltk.sent_tokenize(text, language=lang)
    chunks = []
    chunk = []
    length = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if length + word_count > max_words:
            chunks.append(" ".join(chunk))
            chunk = []
            length = 0
        chunk.append(sentence)
        length += word_count
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# 4. Dynamic chunking (based on token count fluctuation)
def dynamic_chunk(text: str, lang: str, max_tokens: int = 200, min_tokens: int = 50) -> List[str]:
    sentences = nltk.sent_tokenize(text, language=lang)
    chunks = []
    chunk = []
    length = 0
    for sentence in sentences:
        token_count = count_tokens(sentence)
        if length + token_count > max_tokens:
            if length >= min_tokens:
                chunks.append(" ".join(chunk))
                chunk = [sentence]
                length = token_count
            else:
                chunk.append(sentence)
                length += token_count
        else:
            chunk.append(sentence)
            length += token_count
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# 5. Delimiter-based chunking
def delimiter_chunk(text: str, delimiter: str = "\n\n") -> List[str]:
    return [chunk.strip() for chunk in text.split(delimiter) if chunk.strip()]



