from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
# from src.rag.document_loader import Loader
# from src.rag.vectorDB_retriever import VectorDB
# from src.rag.conversation_rag import Conversation_RAG

# Enum for supported model names
class ModelName(str, Enum):
    GEMNINI_1_5_FLASH = "gemini-1.5-flash"  # Google Gemini model
    GPT4_O_MINI = "gpt-4o-mini"             # OpenAI GPT-4o-mini model

# Input schema for a question-answering request
class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")  # User's question
    session_id: str = Field(
        default=None,
        title="Optional session ID. If not provided, one will be generated.",
    )
    model: ModelName = Field(
        default=ModelName.GEMNINI_1_5_FLASH,
        title="Model to use for answering the question",
    )
    # Pydantic config for advanced options (not used here)
    model_config = {"protected_namespaces": ()}

# Output schema for a question-answering response
class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")  # Model's answer
    session_id: str = Field(..., title="Session ID for the conversation")  # Session tracking
    model: str = Field(..., title="Model used to answer the question")     # Model name
    model_config = {"protected_namespaces": ()}

# Metadata for a document in the system
class DocumentInfo(BaseModel):
    id: int                      # Unique document ID
    filename: str                # Name of the uploaded file
    upload_timestamp: datetime   # When the file was uploaded

# Request schema for deleting a file by ID
class DeleteFileRequest(BaseModel):
    file_id: int                 # ID of the file to delete

# Example: How to build a RAG chain (commented out)
# def build_rag_chain(llm, data_dir, data_type):
#     doc_split = Loader(file_types=data_type).load_dir(data_dir, workers=4)
#     retriever = VectorDB(documents=doc_split).get_retriever(llm=llm)
#     rag_chain = Conversation_RAG(llm).get_chain(retriever)
#     return rag_chain
