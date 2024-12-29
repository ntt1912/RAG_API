import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from src.base_llms.llm_model import get_llm
from src.rag.vectorDB_retriever import VectorDB
from dotenv import load_dotenv

load_dotenv()


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question in Vietnamese which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

qa_system_prompt = """Hãy trở thành một trợ lý trả lời câu hỏi. Sử dụng thông tin sau đây để trả lời câu hỏi, hãy trả lời vào đúng trọng tâm của câu hỏi.
Đừng trả lời là dựa trên văn bản mà đi thẳng vào câu trả lời chính.
Nếu không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.
{context}
"""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

class Conversation_RAG:
    def __init__(self, model_name="gpt-4o-mini") -> None:
        self.model_name = model_name
        self.api_key = self._get_api_key()  
        self.llm = get_llm(api_key=self.api_key, model_name=self.model_name) 
        self.retriever = self._get_retriever_for_rag() 
        self.prompt = rag_prompt
        self.contextualize_q_prompt = contextualize_q_prompt

    def _get_api_key(self):
        if self.model_name == "gpt-4o-mini":
            return os.getenv("OPENAI_API_KEY")
        else:
            return os.getenv("GOOGLE_API_KEY")
    
    def _get_retriever_for_rag(self):
        vector_db = VectorDB()
        retriever = vector_db.get_retriever(llm=self.llm)
        return retriever   
        

    def get_chain(self):
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain
