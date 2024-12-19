from typing import Union, List
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereRerank
from langchain.docstore.document import Document
import os, sqlite3
from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv("COHERE_API_KEY")


class VectorDB:
    def __init__(self,
                 documents=None,
                 vector_db: Union[Chroma, FAISS] = Chroma(
                        persist_directory="./src/rag/vector_db",
                        embedding_function=HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert"),
                 ),
                 file_id: int = None,
                 db_path="./src/rag/vector_db/chroma.sqlite3",  # Đường dẫn tới file SQLite
                 ) -> None:
        self.vector_db = vector_db
        self.documents = self._load_documents_from_db(db_path)
        self.bm25 = self._build_bm25Retriever(self.documents) if self.documents else None
        self.reranker = self._build_reranker()

    def _load_documents_from_db(self, db_path):
        """Trích xuất nội dung và metadata từ SQLite và chuyển thành danh sách Document"""
        Chunks = []
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Truy vấn dữ liệu từ bảng
        cursor.execute("SELECT string_value, key, int_value FROM embedding_metadata")
        rows = cursor.fetchall()
        
        # Duyệt qua các dòng kết quả và nhóm metadata lại với từng document
        metadata_dict = {}
        for row in rows:
            string_value, key, int_value = row
            
            if key == "source":
                metadata_dict["source"] = string_value
            elif key == "file_id":
                metadata_dict["file_id"] = int_value
            
            # Chỉ thêm Document nếu page_content (string_value) là chuỗi hợp lệ
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
        return BM25Retriever.from_documents(documents=documents, k=10)
        

    def _build_reranker(self):
        return CohereRerank(model="rerank-multilingual-v3.0", cohere_api_key=api_key,top_n=5)
    
    
    def build_db_and_indexing(self,documents, file_id)-> bool:
        try: 

            for chunk in documents:
                chunk.metadata["file_id"] = file_id

            self.vector_db.add_documents(documents=documents, file_id=file_id)
            return True
        except Exception as e:
            print(f"Error indexing document: {e}")
            return False
        
    def delete_doc_from_chroma(self,file_id):
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
                      search_kwargs: dict = {"k": 10},
                      llm=None  
                      ):
        base_retriever = self.vector_db.as_retriever(search_type=search_type,
                                              search_kwargs=search_kwargs)        

        if llm is not None:
            vector_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm,
            )
        else:
            vector_retriever = base_retriever

         # Kiểm tra bm25_retriever
        if self.bm25:
            retrievers = [vector_retriever, self.bm25]
            weights = [0.5, 0.5]
        else:
            retrievers = [vector_retriever]
            weights = [1.0]   

        # Tạo EnsembleRetriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=weights,
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