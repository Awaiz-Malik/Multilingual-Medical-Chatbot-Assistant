import os
import subprocess
from pinecone import Pinecone
import parsers, embeddings
from chunkers import chunking
from response_retriever import medical_expert_response

PINECONE_API_KEY = "pcsk_6rDter_H7ofP7uZadDpWiQonwHAVm76fFbmhPvrWEnqAeRZfsv6kRSnZNo5wumHRGiq5KT"

# Downloading dataset to directory
os.makedirs('/Users/Lenovo/Downloads/Awaiz_Temp/Assignment_3/dataset/urdu', exist_ok=True)
os.makedirs('/Users/Lenovo/Downloads/Awaiz_Temp/Assignment_3/dataset/spanish', exist_ok=True)

commands = [
    'curl -L -o "C:/Users/Lenovo/Downloads/Awaiz_Temp/Assignment_3/dataset/urdu/rabies.pdf" https://www.immunize.org/wp-content/uploads/vis/urdu_rabies.pdf',
    'curl -L -o "C:/Users/Lenovo/Downloads/Awaiz_Temp/Assignment_3/dataset/urdu/rsvi.pdf" https://www.immunize.org/wp-content/uploads/vis/urdu_iis_rsv.pdf',
    'curl -L -o "C:/Users/Lenovo/Downloads/Awaiz_Temp/Assignment_3/dataset/urdu/heart.pdf" https://www.fda.gov/media/154604/download',
    'curl -L -o "C:/Users/Lenovo/Downloads/Awaiz_Temp/Assignment_3/dataset/urdu/asthma.pdf" https://www.cdc.gov/asthma/pdfs/AsthmaFAQ-factsheet_ur-PK_508.pdf',
    'curl -L -o "C:/Users/Lenovo/Downloads/Awaiz_Temp/Assignment_3/dataset/urdu/anxiety.pdf" https://www.fda.gov/media/173108/download?attachment',
    'curl -L -o "C:/Users/Lenovo/Downloads/Awaiz_Temp/Assignment_3/dataset/spanish/xray.pdf" https://www.cda.org/wp-content/uploads/xrays_spanish.pdf',
    'curl -L -o "C:/Users/Lenovo/Downloads/Awaiz_Temp/Assignment_3/dataset/spanish/iud.pdf" https://www.reproductiveaccess.org/wp-content/uploads/2024/05/2025-01-Copper-IUD-User-Guide-Spanish_Final.pdf',
    'curl -L -o "C:/Users/Lenovo/Downloads/Awaiz_Temp/Assignment_3/dataset/spanish/infection.pdf" https://www.immunize.org/wp-content/uploads/vis/spanish_ppsv.pdf',
    'curl -L -o "C:/Users/Lenovo/Downloads/Awaiz_Temp/Assignment_3/dataset/spanish/badbreath.pdf" https://www.cda.org/wp-content/uploads/bad_breath_spanish.pdf',
    'curl -L -o "C:/Users/Lenovo/Downloads/Awaiz_Temp/Assignment_3/dataset/spanish/bloodclot.pdf" https://www.med.umich.edu/1libr/VTEprevention/PreventingClotsWhileHospitalized-SPN.pdf'
    
]

for cmd in commands:
    subprocess.run(cmd, shell=True, check=True)
    
# Documents parsing

spanish_iud = parsers.doc_parse("dataset/spanish/iud.pdf")
spanish_xray = parsers.doc_parse2("dataset/spanish/xray.pdf")
spanish_blood_clot = parsers.doc_parse("dataset/spanish/bloodclot.pdf")
spanish_bad_breath = parsers.doc_parse("dataset/spanish/badbreath.pdf")
spanish_infection = parsers.doc_parse("dataset/spanish/infection.pdf")
urdu_anxiety = parsers.doc_parse("dataset/urdu/anxiety.pdf")
urdu_rabies = parsers.doc_parse1("dataset/urdu/rabies.pdf")
urdu_asthma = parsers.doc_parse("dataset/urdu/asthma.pdf")
urdu_heart = parsers.doc_parse("dataset/urdu/heart.pdf")
urdu_rsvi = parsers.doc_parse("dataset/urdu/rsvi.pdf")

# Create chunking from document texts

documents = {
    "spanish_xray": spanish_xray,
    "spanish_iud": spanish_iud,
    "spanish_blood_clot": spanish_blood_clot,
    "spanish_bad_breath": spanish_bad_breath,
    "spanish_infection": spanish_infection,
    "urdu_anxiety": urdu_anxiety,
    "urdu_rabies": urdu_rabies,
    "urdu_asthma": urdu_asthma,
    "urdu_heart": urdu_heart,
    "urdu_rsvi": urdu_rsvi
}

chunked_documents = chunking(documents)

# Create embeddings

vector_store_input, model = embeddings.embed(chunked_documents)

# Store in pinecone vector database

pinecone_host_link = "https://multilingual1024-i22y2e5.svc.aped-4627-b74a.pinecone.io"
pinecone_index_name = "multilingual1024"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=pinecone_host_link)
if pinecone_index_name not in pc.list_indexes().names(): # Check if index exists
    raise Exception(f"Index may not exist.")
# index.upsert(vectors=vector_store_input, namespace="medical_dataset")

# Retrieval of Final response from llm

query = "اضطراب کے امراض کی اہم اقسام کونسی ہیں؟ "

response = medical_expert_response(query, index, model)

print(response)