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

def load_documents(db="chroma", recursive=True):
    """
    Load all PDF document and insert the chunks into a database
    """
    pdf_files = glob(str(FULL_DOCS_PATH.joinpath("**/*.pdf")), recursive=recursive)
    data = [PyMuPDFLoader(FULL_DOCS_PATH.joinpath(pdf).as_posix()).load() for pdf in pdf_files]
    docs = [doc for d in data for doc in text_splitter.split_documents(d)]
    if db == "chroma":
        return ChromaDB(docs, save=True)
    elif db == "pinecone":
        return PineconeDB(docs)
    raise ValueError("Invalid db choice")

