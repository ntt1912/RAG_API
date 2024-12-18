import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from src.base.llm_model import get_hf_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA
from src.rag.db_utils import insert_application_logs, get_chat_history
from src.chat.main import InputChat
from src.chat.main import build_chat_chain
from dotenv import load_dotenv
import uuid
import shutil
import logging

logging.basicConfig(filename='app.log', level=logging.INFO)


load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
# API_KEY = os.getenv("OPENAI_API_KEY")
llm = get_hf_llm(api_key=API_KEY)
# llm = get_hf_llm(temperature=0.4)

iot_docs = "./data_source/IoT"

# --------- Chains----------------

iot_chain = build_rag_chain(llm, data_dir=iot_docs, data_type="pdf")
chat_chain = build_chat_chain(llm, 
                              history_folder="./chat_histories",
                              max_history_length=8)


# --------- App - FastAPI ----------------

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
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
    session_id = inputs.session_id or str(uuid.uuid4())
    logging.info(f"Session ID: {session_id},User_query : {inputs.question}")
    chat_history = get_chat_history(session_id) 
    answer = iot_chain.invoke({
        "input": inputs.question,
        "chat_history": chat_history
    })['answer']

    insert_application_logs(session_id, inputs.question, answer, inputs.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return OutputQA(answer=answer, session_id=session_id,model= inputs.model.value)

@app.post("/chat", response_model=OutputQA)
async def chat(inputs: InputChat):
    answer = chat_chain.invoke(
            {"human_input": inputs.human_input},  
            {"configurable": {"session_id": inputs.session_id}}  
    )
    return {"answer": answer}

# --------- Langserve Routes - Playground ----------------
add_routes(app, 
           iot_chain, 
           playground_type="default",
           path="/IoT")

add_routes(app,
           chat_chain,
           enable_feedback_endpoint=False,
           enable_public_trace_link_endpoint=False,
           playground_type="default",
           path="/chat")