import os
import tomllib
import uuid
from pinecone import Pinecone
from pathlib import Path
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Pinecone as Pinecone_LC
from langchain_community.vectorstores import Chroma

with open(Path().resolve().parent / "config/chatbot.toml", "rb") as c:
    config = tomllib.load(c)

key = os.environ["PINECONE_API_KEY"] = os.getenv("pinecone_key")
pc = Pinecone(api_key=key )

ABSOLUTE_PATH = Path().resolve().parent

class PineconeDB():
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

class ChromaDB():
    def __init__(self, docs, collection_name = "chroma_db", save=False):
        self.hf_embeddings = HuggingFaceInstructEmbeddings()
        assert len(docs) > 0, "Number of docs cannot be empty!"
        db = Chroma(persist_directory = ABSOLUTE_PATH.joinpath("chromadb/").as_posix(), 
                     collection_name=collection_name,
                     embedding_function=self.hf_embeddings)
        if db._collection.count() == 0:
            db = Chroma.from_documents(
                documents = docs, 
                ids = [str(uuid.uuid4()) for _ in range(len(docs))],
                embedding=self.hf_embeddings, 
                persist_directory= ABSOLUTE_PATH.joinpath("chromadb/").as_posix(),
                collection_name=collection_name
            )
        else:
            db.add_texts(
                texts = [doc.page_content for doc in docs],
                metadatas = [doc.metadata for doc in docs],
                ids = [str(uuid.uuid4()) for _ in range(len(docs))],
            )
        print(db._collection.count())

        # Save the Chroma database to disk
        if save:
            db.persist()
        self.db = db
        self.save = save

    def query(self, query_texts, max_results=3):
        results = self.db.similarity_search(
            query_texts, 
            k=max_results)
        return " ".join([text.page_content for text in results])

    def add(self, docs):
        assert len(docs) > 0, "Number of docs cannot be empty!"
        self.db.add_texts(
            texts = [doc.page_content for doc in docs],
            metadatas = [doc.metadata for doc in docs],
            ids = [str(uuid.uuid4()) for _ in range(len(docs))],
        )
        if self.save:
            self.db.persist()

    def get_num_documents(self):
        return self.db._collection.count()
    
    def delete_database(self):
        self.db.delete_collection()
