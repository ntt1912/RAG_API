from typing import Union
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
# from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereRerank
from langchain.docstore.document import Document
import os, sqlite3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("COHERE_API_KEY")

class VectorDB:
    """
    Class for managing vector database, document retrieval, and hybrid search (vector + BM25 + reranking).
    Handles loading documents from SQLite, indexing, deletion, and building advanced retrievers.
    """
    def __init__(self,
                 documents=None,
                 vector_db: Union[Chroma, FAISS] = Chroma(
                        persist_directory="./src/rag/vector_db",
                        embedding_function=OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1024, api_key=openai_api_key),
                 ),
                 file_id: int = None,
                 db_path="./src/rag/vector_db/chroma.sqlite3",  # Path to SQLite file
                 ) -> None:
        # Vector database instance (Chroma or FAISS)
        self.vector_db = vector_db
        # Load all documents from the SQLite database
        self.documents = self._load_documents_from_db(db_path)
        # Build BM25 retriever for keyword-based search
        self.bm25 = self._build_bm25Retriever(self.documents) if self.documents else None
        # Build reranker (Cohere)
        self.reranker = self._build_reranker()

    def _load_documents_from_db(self, db_path):
        """
        Extract content and metadata from SQLite and convert to a list of Document objects.
        """
        Chunks = []
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Query all metadata rows
        cursor.execute("SELECT string_value, key, int_value FROM embedding_metadata")
        rows = cursor.fetchall()
        # Group metadata for each document
        metadata_dict = {}
        for row in rows:
            string_value, key, int_value = row
            if key == "source":
                metadata_dict["source"] = string_value
            elif key == "file_id":
                metadata_dict["file_id"] = int_value
            # Only add Document if page_content is a valid string
            if string_value and isinstance(string_value, str):
                Chunks.append(
                    Document(
                        page_content=string_value,
                        metadata={key: metadata_dict.get(key, None) for key in ["source", "file_id"]}
                    )
                )
        conn.close()
        return Chunks

    def _build_bm25Retriever(self, documents):
        """
        Build a BM25 keyword retriever from a list of documents.
        """
        return BM25Retriever.from_documents(documents=documents, k=8)

    def _build_reranker(self):
        """
        Build a Cohere reranker for semantic reranking of retrieved documents.
        """
        return CohereRerank(model="rerank-multilingual-v3.0", cohere_api_key=api_key, top_n=5)
    
    def build_db_and_indexing(self, documents, file_id) -> bool:
        """
        Add documents to the vector database and associate them with a file_id.
        Returns True if successful, False otherwise.
        """
        try:
            for chunk in documents:
                chunk.metadata["file_id"] = file_id
            self.vector_db.add_documents(documents=documents, file_id=file_id)
            return True
        except Exception as e:
            print(f"Error indexing document: {e}")
            return False
    
    def delete_doc_from_chroma(self, file_id):
        """
        Delete all document chunks from Chroma vector DB associated with a given file_id.
        Returns True if successful, False otherwise.
        """
        try:
            docs = self.vector_db.get(where={"file_id": file_id})
            print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")
            self.vector_db._collection.delete(where={"file_id": file_id})
            print(f"Deleted all documents with file_id {file_id}")
            return True
        except Exception as e:
            print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
            return False

    def get_retriever(self, 
                      search_type: str = "similarity", 
                      search_kwargs: dict = {"k": 8},
                      llm=None  
                      ):
        """
        Build a hybrid retriever that combines vector search, BM25, and reranking.
        If an LLM is provided, use MultiQueryRetriever for query expansion.
        Returns a ContextualCompressionRetriever for efficient retrieval.
        """
        # Create base vector retriever
        base_retriever = self.vector_db.as_retriever(search_type=search_type,
                                              search_kwargs=search_kwargs)        
        # Use LLM for multi-query expansion if provided
        if llm is not None:
            vector_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm,
            )
        else:
            vector_retriever = base_retriever
        # Combine vector retriever and BM25 with equal weights
        retrievers = [vector_retriever, self.bm25]
        weights = [0.5, 0.5]
        # Create an ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=weights,
        )
        # Add contextual compression (reranking)
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=ensemble_retriever,
            base_compressor=self.reranker,
        )
        return compression_retriever


