import json
import os
import uuid
import PyPDF2
from glob import glob
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from database import PineconeDB, ChromaDB

ABSOLUTE_PATH = Path().resolve().parent
DOCS_PATH = Path("docs")
FULL_DOCS_PATH = ABSOLUTE_PATH.joinpath(DOCS_PATH)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap  = 20,
    length_function = len,
    is_separator_regex = False,
)

token_splitter = TokenTextSplitter(
    chunk_size=1000, 
    chunk_overlap=10,
    encoding_name="cl100k_base"
)

def generate_or_update_metadata(pdf_files):
    """
    Generates a JSON metadata file to store the required contents for all PDF documents. These include:
    - The filename of the PDF
    - The size in MBs
    - The page count of the document
    """
    metadata_path = FULL_DOCS_PATH.joinpath("metadata.json").as_posix()
    with open(metadata_path, "w") as file:
        output = []
        for pdf in pdf_files:
            metadata = {}
            doc_data = {}
            id = str(uuid.uuid4())
            name = Path(pdf).name
            file_stats = os.stat(pdf)
            size_mb = round(file_stats.st_size / (1024 * 1024), 3)
            page_count = len(PyPDF2.PdfReader(pdf).pages)
            doc_data["filename"] = name
            doc_data["size_mb"] = size_mb
            doc_data["page_count"] = page_count
            metadata["id"] = id
            metadata["content"] = doc_data
            output.append(metadata)
        json.dump(output, file, indent=2)

    ids = {}
    with open(metadata_path, "r") as file:
        metadata = json.load(file)
        for entry in metadata:
            ids.__setitem__(entry["content"]["filename"], entry["id"])
    return ids 

def load_documents(db="chroma", recursive=True):
    """
    Load all PDF document and insert the chunks into a database
    """
    pdf_files = glob(str(FULL_DOCS_PATH.joinpath("**/*.pdf")), recursive=recursive)
    ids = generate_or_update_metadata(pdf_files)
    db_data = {}
    data = [PyMuPDFLoader(FULL_DOCS_PATH.joinpath(pdf).as_posix()).load() for pdf in pdf_files]
    for d, id in zip(data, ids):
        for i, doc in enumerate(text_splitter.split_documents(d)):
            id = f"{ids.get(Path(doc.metadata.get('source')).name)}_{i}"
            db_data.__setitem__(id, doc)
    if db == "chroma":
        return ChromaDB(db_data, save=True)
    elif db == "pinecone":
        return PineconeDB(db_data)
    raise ValueError("Invalid db choice")

