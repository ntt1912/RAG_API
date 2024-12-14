import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from src.base.llm_model import get_hf_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA
from src.chat.main import InputChat
from src.chat.main import build_chat_chain
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
llm = get_hf_llm(api_key=API_KEY)
# llm = get_hf_llm(temperature=0.4)

genai_docs = "./data_source/generative_ai"
ml_docs = "./data_source/machine_learning"

# --------- Chains----------------

genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")
ml_chain = build_rag_chain(llm, data_dir=ml_docs, data_type="html")

chat_chain = build_chat_chain(llm, 
                              history_folder="./chat_histories",
                              max_history_length=6)


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


@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}


@app.post("/machine_learning", response_model=OutputQA)
async def machine_learning(inputs: InputQA):
    answer = ml_chain.invoke(inputs.question)
    return {"answer": answer}

@app.post("/chat", response_model=OutputQA)
async def chat(inputs: InputChat):
    answer = chat_chain.invoke(
            {"human_input": inputs.human_input},  
            {"configurable": {"session_id": inputs.session_id}}  
    )
    return {"answer": answer}

# --------- Langserve Routes - Playground ----------------
add_routes(app, 
           genai_chain, 
           playground_type="default",
           path="/generative_ai")

add_routes(app, 
           ml_chain, 
           playground_type="default",
           path="/machine_learning")


add_routes(app,
           chat_chain,
           enable_feedback_endpoint=False,
           enable_public_trace_link_endpoint=False,
           playground_type="default",
           path="/chat")