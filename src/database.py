import os
import tomllib
from pinecone import Pinecone
from pathlib import Path
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Pinecone as Pinecone_LC

with open(Path().resolve().parent / "config/chatbot.toml", "rb") as c:
    config = tomllib.load(c)

key = os.environ["PINECONE_API_KEY"] = os.getenv("pinecone_key")
pc = Pinecone(api_key=key )

class Database():
    def __init__(self, docs):
        self.hf_embeddings = HuggingFaceInstructEmbeddings()
        self.index_name = "test-index"
        self.db = Pinecone_LC.from_documents(docs, 
                                             self.hf_embeddings, 
                                             index_name=self.index_name)

    def query(self, query, max_results=3):
        results = self.db.similarity_search(query, k=max_results)
        return " ".join([text.page_content for text in results])

    def add(self, docs):
        index = pc.Index(self.index_name)
        vectorstore = Pinecone(index, self.hf_embedding.embed_query, "text")
        vectorstore.add_texts(docs)