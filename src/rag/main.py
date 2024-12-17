from pydantic import BaseModel, Field
from src.rag.file_loader import Loader
from src.rag.vectorstore import VectorDB
from src.rag.offline_rag import Offline_RAG
from enum import Enum

class ModelName(str,Enum):
    GPT3_5= "gpt3.5-turbo"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"



class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")
    session_id: str = Field(default=None, title="Optional session ID. If not provided, one will be generated.")
    model: ModelName = Field(default=ModelName.GEMINI_1_5_FLASH, title="Model to use for answering the question")
    model_config = {
        'protected_namespaces': ()
    }

class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")
    session_id: str = Field(..., title="Session ID for the conversation")
    model: ModelName
    model_config = {
        'protected_namespaces': ()
    }


def build_rag_chain(llm, data_dir, data_type):
    doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2)
    retriever = VectorDB(documents = doc_loaded).get_retriever()
    rag_chain = Offline_RAG(llm).get_chain(retriever)
    return rag_chain
