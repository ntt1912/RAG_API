from pydantic import BaseModel, Field
from datetime import datetime
# from src.rag.document_loader import Loader
# from src.rag.vectorDB_retriever import VectorDB
# from src.rag.conversation_rag import Conversation_RAG

class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")
    session_id: str = Field(
        default=None,
        title="Optional session ID. If not provided, one will be generated.",
    )
    model_config = {"protected_namespaces": ()}


class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")
    session_id: str = Field(..., title="Session ID for the conversation")
    model: str = Field(..., title="Model used to answer the question")
    model_config = {"protected_namespaces": ()}

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: int


# def build_rag_chain(llm, data_dir, data_type):
#     doc_split = Loader(file_types=data_type).load_dir(data_dir, workers=4)
#     retriever = VectorDB(documents=doc_split).get_retriever(llm=llm)
#     rag_chain = Conversation_RAG(llm).get_chain(retriever)
#     return rag_chain
