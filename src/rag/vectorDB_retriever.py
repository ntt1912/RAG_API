from typing import Union, List, Any
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereRerank
import os
from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv("COHERE_API_KEY")

class VectorDB:
    def __init__(self,
                 documents=None,
                 vector_db: Union[Chroma, FAISS] = Chroma,
                 embedding_model: str = 'keepitreal/vietnamese-sbert',
                 persist_directory: str = "./src/rag/vector_db",
                 ids: List[str] = None  
                 ) -> None:
        # Sử dụng Sentence-BERT làm model embedding
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_db = vector_db
        self.persist_directory = persist_directory
        self.db = self._build_db(documents, ids) if documents else None
        self.bm25= self._build_bm25Retriever(documents) if documents else None
        self.reranker = self._build_reranker()
        self.ids = ids 

    def _build_db(self, documents, ids):
        db = self.vector_db.from_documents(documents=documents, 
                                           embedding=self.embedding,
                                           persist_directory=self.persist_directory,
                                           ids=ids)
        return db
    def _build_bm25Retriever(self,documents):
        bm_25_retriever = BM25Retriever.from_documents(documents=documents,k=10)
        return bm_25_retriever
    
    def _build_reranker(self):
        compressor = CohereRerank(model="rerank-multilingual-v3.0", cohere_api_key=api_key,top_n=5)
        return compressor


    def get_retriever(self, 
                      search_type: str = "similarity", 
                      search_kwargs: dict = {"k": 10},
                      llm=None  
                      ):
        base_retriever = self.db.as_retriever(search_type=search_type,
                                              search_kwargs=search_kwargs)
        bm25_retriever = self.bm25
        

        if llm is not None:
            vector_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm,
            )
        else:
            vector_retriever = base_retriever

        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )

        compression_retriever = ContextualCompressionRetriever(
            base_retriever=ensemble_retriever,
            base_compressor=self.reranker,
        )

        return compression_retriever



   

#     def __init__(self,
#                  documents=None,
#                  vector_db: Union[Chroma, FAISS] = Chroma,
#                  embedding_model: str = 'text-embedding-ada-002',  # OpenAI embedding model
#                  openai_api_key: str = api_key, 
#                  ) -> None:
#         # Sử dụng OpenAIEmbeddings
#         self.embedding = OpenAIEmbeddings(model=embedding_model, openai_api_key=openai_api_key)