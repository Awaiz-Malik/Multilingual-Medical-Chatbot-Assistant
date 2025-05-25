import fitz

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