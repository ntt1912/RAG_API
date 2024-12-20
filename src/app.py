import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.base_llms.llm_model import get_llm
from src.rag.pydantic_utils import InputQA, OutputQA, DocumentInfo, DeleteFileRequest
from src.rag.db_utils import insert_application_logs, get_rag_history
from src.chatchit.main import InputChat, OutputChat
from src.chatchit.main import build_chat_chain
from src.rag.db_utils import insert_document_record, delete_document_record, get_all_documents
from src.rag.document_loader import Loader
from src.rag.vectorDB_retriever import VectorDB
from src.rag.conversation_rag import Conversation_RAG
from dotenv import load_dotenv
import uuid
import shutil
import logging


logging.basicConfig(filename='app.log', level=logging.INFO)


load_dotenv()

MODEL_NAME = "gemini-1.5-flash"
if MODEL_NAME == "gemini-1.5-flash":
    API_KEY = os.getenv("GOOGLE_API_KEY")
else:
    API_KEY = os.getenv("OPENAI_API_KEY")

llm = get_llm(api_key=API_KEY, model_name=MODEL_NAME)
# llm = get_hf_llm(temperature=0.4)
#--------- Retiever ----------------


# iot_docs = "./data_source/IoT"

# --------- Chains----------------

# iot_chain = build_rag_chain(llm, data_dir=iot_docs, data_type=["pdf", "docx"])
chat_chain = build_chat_chain(llm, 
                              history_folder="./chat_histories",
                              max_history_length=100)


# --------- App - FastAPI ----------------

app = FastAPI(
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# --------- Routes - FastAPI ----------------

@app.get("/check")
async def check():
    return {"status": "ok"}


@app.post("/IoT", response_model=OutputQA)
async def IoT(inputs: InputQA):
    questions = inputs.question
    session_id = inputs.session_id or str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}, User Query: {inputs.question}, Model: {inputs.model.value}")
    chat_history = get_rag_history(session_id) 
    iot_chain = Conversation_RAG(model_name=inputs.model.value).get_chain()
    answer = iot_chain.invoke({
        "input": questions,
        "chat_history": chat_history
    })['answer']

    insert_application_logs(session_id, inputs.question, answer,inputs.model.value) 
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return OutputQA(answer=answer, session_id=session_id, model=inputs.model)

@app.post("/chat", response_model=OutputChat)
async def chat(inputs: InputChat):
    question=inputs.human_input
    session_id = inputs.session_id or str(uuid.uuid4())
    answer = chat_chain.invoke(
            {"human_input": question},  
            {'configurable': {'session_id': session_id}}
    )
    return OutputChat(answer=answer, session_id=session_id, model=MODEL_NAME)

@app.post("/upload-doc")
async def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")

    temp_file_path = f"temp_{file.filename}"

    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = insert_document_record(file.filename)
        doc_split = Loader(file_types=[file_extension[1:]]).load_dir(temp_file_path)
        vector_db = VectorDB(file_id=file_id)
        success = vector_db.build_db_and_indexing(documents=doc_split,file_id=file_id)

        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/view-docs", response_model=list[DocumentInfo])
async def view_documents():
    return get_all_documents()

@app.post("/delete-doc")
async def delete_document(request: DeleteFileRequest):
    # Delete from Chroma
    chroma_delete_success = VectorDB().delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        # If successfully deleted from Chroma, delete from our database
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}

