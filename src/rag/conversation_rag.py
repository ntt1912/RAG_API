# Standard library imports
import os

# LangChain and project-specific imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from src.base_llms.llm_model import get_llm
from src.rag.vectorDB_retriever import VectorDB
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# System prompt for contextualizing user questions
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question in Vietnamese which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# Prompt template for question contextualization (makes user question standalone)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# System prompt for the RAG assistant (Vietnamese)
qa_system_prompt = """
Hãy trở thành một trợ lý trả lời câu hỏi. Sử dụng thông tin sau đây để trả lời câu hỏi, hãy trả lời vào đúng trọng tâm của câu hỏi.
Đừng trả lời là dựa trên văn bản mà đi thẳng vào câu trả lời chính.
Nếu không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.
{context}
"""

# Prompt template for the RAG chain (includes context and chat history)
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

class Conversation_RAG:
    """
    Main class for handling conversational Retrieval-Augmented Generation (RAG).
    Handles LLM selection, retriever setup, and chain construction for chat-based RAG.
    """
    def __init__(self, model_name=None) -> None:
        # Model name (e.g., 'gpt-4o-mini' or other supported models)
        self.model_name = model_name
        # Get API key based on model type
        self.api_key = self._get_api_key()  
        # Initialize the language model
        self.llm = get_llm(api_key=self.api_key, model_name=self.model_name) 
        # Set up the retriever (vector database)
        self.retriever = self._get_retriever_for_rag() 
        # Prompt for the RAG chain
        self.prompt = rag_prompt
        # Prompt for contextualizing user questions
        self.contextualize_q_prompt = contextualize_q_prompt

    def _get_api_key(self):
        """
        Select the correct API key based on the model name.
        """
        if self.model_name == "gpt-4o-mini":
            return os.getenv("OPENAI_API_KEY")
        else:
            return os.getenv("GOOGLE_API_KEY")
    
    def _get_retriever_for_rag(self):
        """
        Initialize and return a retriever from the vector database, using the LLM.
        """
        vector_db = VectorDB()
        return vector_db.get_retriever(llm=self.llm)

    def get_chain(self):
        """
        Build and return the full RAG chain for conversational question answering.
        This includes a history-aware retriever and a document QA chain.
        """
        # Create a retriever that is aware of chat history
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )
        # Create a chain that stuffs retrieved documents and answers the question
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)

        # Combine retriever and QA chain into a full RAG chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain
