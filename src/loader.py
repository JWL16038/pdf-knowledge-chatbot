import fitz
from pathlib import Path
from langchain.document_loaders import PyMuPDFLoader

ABSOLUTE_PATH = Path().resolve()
DOCS_PATH = Path("docs")
FULL_DOCS_PATH = ABSOLUTE_PATH.joinpath(DOCS_PATH)

loader = PyMuPDFLoader("docs/comp307 test 22.pdf")
data = loader.load()

for d in data:
    print(d.page_content)