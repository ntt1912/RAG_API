from typing import Union, List
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import OpenAIEmbeddings

# import os
# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# class VectorDB:
#     def __init__(self,
#                  documents = None,
#                  vector_db: Union[Chroma, FAISS] = Chroma,
#                  embedding = HuggingFaceEmbeddings(),
#                  ) -> None:
        
#         self.vector_db = vector_db
#         self.embedding = embedding
#         self.db = self._build_db(documents)

#     def _build_db(self, documents):
#         db = self.vector_db.from_documents(documents=documents, 
#                                           embedding=self.embedding)
#         return db

#     def get_retriever(self, 
#                       search_type: str = "similarity", 
#                       search_kwargs: dict = {"k": 3}
#                       ):
#         retriever = self.db.as_retriever(search_type=search_type,
#                                          search_kwargs=search_kwargs)
#         return retriever

class VectorDB:
    def __init__(self,
                 documents=None,
                 vector_db: Union[Chroma, FAISS] = Chroma,
                 embedding_model: str = 'sentence-transformers/all-MiniLM-L12-v2',  # default to Sentence-BERT
                 ) -> None:
        # Sử dụng Sentence-BERT làm model embedding
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_db = vector_db
        self.db = self._build_db(documents)

    def _build_db(self, documents):
        db = self.vector_db.from_documents(documents=documents, 
                                           embedding=self.embedding)
        return db

    def get_retriever(self, 
                      search_type: str = "similarity", 
                      search_kwargs: dict = {"k": 4},
                      llm=None  
                      ):
        base_retriever = self.db.as_retriever(search_type=search_type,
                                              search_kwargs=search_kwargs)
        if llm is not None:
            retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm,
                search_kwargs={"k": search_kwargs["k"]}
            )
        else:
            retriever = base_retriever

        return retriever

#     def __init__(self,
#                  documents=None,
#                  vector_db: Union[Chroma, FAISS] = Chroma,
#                  embedding_model: str = 'text-embedding-ada-002',  # OpenAI embedding model
#                  openai_api_key: str = api_key, 
#                  ) -> None:
#         # Sử dụng OpenAIEmbeddings
#         self.embedding = OpenAIEmbeddings(model=embedding_model, openai_api_key=openai_api_key)