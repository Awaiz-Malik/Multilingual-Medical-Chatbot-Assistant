import re
from typing import List
from langchain_text_splitters import CharacterTextSplitter

def chunk_questions_answers(text):
    question_matches = list(re.finditer(r'¿[^?]+\?', text))
    chunks = []

    for i, match in enumerate(question_matches):
        question = match.group().strip()
        start = match.end()
        end = question_matches[i + 1].start() if i + 1 < len(question_matches) else len(text)
        answer = text[start:end].strip()
        if answer:
            chunks.append(f"{question} {answer}")

    return chunks

def chunk_numbered_sections(text):
    pattern = r'(?:^|\n)\s*(\d+)\s+[^\n]+'
    matches = list(re.finditer(pattern, text))
    chunks = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        number_match = re.match(r'\s*(\d+)\s+(.*)', section_text)
        if number_match:
            section_number = number_match.group(1).strip()
            title = number_match.group(2).split('\n')[0].strip()  # First line after number
            content = section_text[len(number_match.group(0)):].strip()
            chunks.append(f"{section_number}. {title} {content}")
    return chunks

def chunk_by_number(text):
    pattern = re.compile(r'(?=^\s*\.\d+\s.*)', re.MULTILINE)
    matches = list(pattern.finditer(text))

    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        chunks.append(chunk)

    return chunks

def chunk_urdu(text):
    question_matches = list(re.finditer(r'.*?؟', text))

    chunks = []
    for i in range(len(question_matches)):
        question_span = question_matches[i].span()
        question = question_matches[i].group().strip()
        answer_start = question_span[1]

        if i + 1 < len(question_matches):
            answer_end = question_matches[i + 1].start()
        else:
            answer_end = len(text)

        answer = text[answer_start:answer_end].strip()
        chunk = f"{question} {answer}"
        chunks.append(chunk)

    return chunks

def character_text_splitter(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separator= "\n")
    return splitter.split_text(text)



chunked_documents = {}

def chunking(documents):
    # Define chunking strategy map
    chunking_strategies = {
        "spanish_xray": chunk_questions_answers,
        "spanish_iud": chunk_questions_answers,
        "spanish_blood_clot": chunk_questions_answers,
        "spanish_bad_breath": lambda text: character_text_splitter(text, chunk_size=1000, overlap=230),
        "spanish_infection": chunk_numbered_sections,
        "urdu_anxiety": chunk_urdu,
        "urdu_rabies": chunk_by_number,
        "urdu_asthma": chunk_urdu,
        "urdu_heart": lambda text: character_text_splitter(text, chunk_size=544, overlap=200),
        "urdu_rsvi": lambda text: character_text_splitter(text, chunk_size=1300, overlap=0)
    }
    
    # Apply the appropriate chunking function in one line
    chunked_documents = {name: chunking_strategies[name](text) for name, text in documents.items()}
    
    return chunked_documents
