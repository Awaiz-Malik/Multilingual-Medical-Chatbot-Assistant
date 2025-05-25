from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-large')

source_links = {
    "urdu_rabies": "https://www.immunize.org/wp-content/uploads/vis/urdu_rabies.pdf",
    "urdu_anxiety": "https://www.fda.gov/media/173108/download?attachment",
    "urdu_asthma": "https://www.cdc.gov/asthma/pdfs/AsthmaFAQ-factsheet_ur-PK_508.pdf",
    "urdu_heart": "https://www.fda.gov/media/154604/download",
    "urdu_rsvi": "https://www.immunize.org/wp-content/uploads/vis/urdu_iis_rsv.pdf",
    "spanish_xray": "https://www.cda.org/wp-content/uploads/xrays_spanish.pdf",
    "spanish_iud": "https://www.cda.org/wp-content/uploads/iud_spanish.pdf",
    "spanish_infection": "https://www.immunize.org/wp-content/uploads/vis/spanish_ppsv.pdf",
    "spanish_bad_breath": "https://www.cda.org/wp-content/uploads/bad_breath_spanish.pdf",
    "spanish_blood_clot": "https://www.med.umich.edu/1libr/VTEprevention/PreventingClotsWhileHospitalized-SPN.pdf"
}

vector_store_input = []

def embed(chunked_documents):
    i = 1
    for doc_name, chunks in chunked_documents.items():
        source_link = source_links.get(doc_name, "")

        for chunk in chunks:
            vector_store_input.append({
                "id": f"{i}",
                "values": model.encode(chunk),
                "metadata": {
                    "text": chunk,
                    "source_link": source_link,
                }
            })
            i += 1
    
    return vector_store_input, model