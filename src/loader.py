import json
import os
import uuid
import PyPDF2
import fitz
import pycld2 as cld2
from glob import glob
import shutil
from pathlib import Path
import fitz
import ocrmypdf
from ocrmypdf.exceptions import TaggedPDFError, EncryptedPdfError
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

def detect_languages(pdf):
    """
    Detect all languages (as codes) that are present in the document. Languages which are unknown (un) will be ignored.
    """
    languages = []
    with fitz.open(pdf) as doc:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            _, _, _, detected_langs = cld2.detect(text,  returnVectors=True)
            if detected_langs:
                for lang in detected_langs:
                    if lang[3] != "un":
                        languages.append(lang[3])
    if not languages:
        return None
    languages = list(set(languages))
    return languages

def check_searchable(pdf):
    """
    Check if the entire PDF is searchable, that being all pages does not contain any searchable text.
    """
    searchable_page_count = 0
    with fitz.open(pdf) as doc:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                searchable_page_count += 1
        if searchable_page_count > 0:
            return True
        return False

def generate_or_update_metadata(pdf_files):
    """
    Generates a JSON metadata file to store the required contents for all PDF documents. The JSON will include:
    - The filename of the PDF
    - The size in MBs
    - The page count of the document
    
    Steps:
    - Checks if the JSON file is not malformed
    - Detect any documents that are still in pdf_files (skip list)
    - Detect any documents that no longer exist in pdf_files (delete list) 
    - Parses the JSON file for new documents and delete any old ones. 
    - Returns all valid IDs for each document in the JSON.
    """
    metadata_path = FULL_DOCS_PATH.joinpath("metadata.json").as_posix()
    skip_list = [] # If the pdf exists in the pdf_files list and in JSON - skip
    remove_list = [] # If the pdf doesn't exist in the pdf_files list but still in JSON - delete
    metadata_json = []
    # Check if the metadata exists and not an empty JSON, then check if the entries in pdf_files exist in the JSON. Also check for 'dead' entries in the JSON where the document has been removed and still exists in the JSON.
    if os.path.exists(metadata_path) and os.path.getsize(metadata_path) != 0:
        with open(metadata_path, "r") as file:
            try:
                metadata_json = json.load(file)
                for pdf in pdf_files:
                    name = Path(pdf).name
                    results = list(filter(lambda x:x["content"]["filename"]==name,metadata_json))
                    if len(results) > 0:
                        skip_list.append(results[0]["content"]["filename"])
                pdf_filenames = [Path(x).name for x in pdf_files]
                for x in metadata_json:
                    if x["content"]["filename"] not in pdf_filenames:
                        remove_list.append(x["content"]["filename"])
            except ValueError:
                os.remove(metadata_path)

    with open(metadata_path, "w") as file:
        # Delete any entries that are in the remove list
        if remove_list:
            metadata_json = [x for x in metadata_json if x["content"]["filename"] not in remove_list]

        # Parses all new documents in the document directory
        for pdf in pdf_files:
            metadata = {}
            doc_data = {}
            name = Path(pdf).name
            if name in skip_list:
                continue
            id = str(uuid.uuid4())
            file_stats = os.stat(pdf)
            size_mb = round(file_stats.st_size / (1024 * 1024), 3)
            page_count = len(PyPDF2.PdfReader(pdf).pages)
            searchable = check_searchable(pdf)
            langauges = detect_languages(pdf)
            doc_data["filename"] = name
            doc_data["size_mb"] = size_mb
            doc_data["page_count"] = page_count
            doc_data["searchable"] = searchable
            doc_data["languages"] = langauges
            metadata["id"] = id
            metadata["content"] = doc_data
            metadata_json.append(metadata)
        json.dump(metadata_json, file, indent=2)

    # Loads all valid document ids with the associated filename
    ids = {}
    with open(metadata_path, "r") as file:
        metadata = json.load(file)
        for entry in metadata:
            ids.__setitem__(entry["content"]["filename"], entry["id"])
    return ids 

def ocr_documents(pdf_files):
    """
    Check if the entire PDF is searchable, if not OCR it and save the result as a new OCRed doc
    """
    if not os.path.isdir(FULL_DOCS_PATH.joinpath("ocr")):
        os.mkdir(FULL_DOCS_PATH.joinpath("ocr"))
    if not os.path.isdir(FULL_DOCS_PATH.joinpath("noocr")):
        os.mkdir(FULL_DOCS_PATH.joinpath("noocr"))

    for pdf_path in pdf_files:
        searchable = check_searchable(pdf_path)
        if searchable:
            continue
        new_name = FULL_DOCS_PATH.joinpath(f"ocr/{Path(pdf_path).stem}_ocr.pdf")
        try:
            ocrmypdf.ocr(pdf_path, 
                        new_name,
                        output_type="pdf",
                        deskew=True, 
                        )
        except TaggedPDFError:
            # Detected that the PDF is a Word doc but with no searchable text - force ocr
            ocrmypdf.ocr(pdf_path, 
                        new_name,
                        output_type="pdf",
                        force_ocr=True,
                        deskew=True, 
                        )
        except EncryptedPdfError:
            print("PDF is encrypted, skipping...")
            continue            
        # shutil.move(pdf_path, FULL_DOCS_PATH.joinpath(f"noocr/{Path(pdf_path).stem}.pdf"))
        i = pdf_files.index(pdf_path)
        pdf_files[i] = new_name
    return pdf_files

def load_documents(db="chroma", recursive=True):
    """
    Load all PDF document and insert the chunks into a database
    """
    all_files = glob(str(FULL_DOCS_PATH.joinpath("**/*.pdf")), 
                     recursive=recursive)
    nonocr_files = glob(str(FULL_DOCS_PATH.joinpath("**/noocr/*.pdf")), 
                     recursive=recursive)
    pdf_files = list(set(all_files) - set(nonocr_files))
    pdf_files = ocr_documents(pdf_files)
    db_data = {}
    ids = generate_or_update_metadata(pdf_files)
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
