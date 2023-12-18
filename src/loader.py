from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from database import Database

ABSOLUTE_PATH = Path().resolve()
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

def load_pdf(pdf_path="docs/comp307 test 22.pdf") -> Database:
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    docs = text_splitter.split_documents(data)
    db = Database(docs)
    return db

