from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings

class Database():
    def __init__(self, docs):
        self.hf_embedding = HuggingFaceInstructEmbeddings()
        # Chroma loading
        self.db = Chroma.from_documents(docs, self.hf_embedding)

        # FAISS loading
        # self.db = FAISS.from_documents(docs, self.hf_embedding)

    def query(self, query, max_results=3):
        return self.db.similarity_search(query, k=max_results)
