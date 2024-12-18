from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

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

class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = rag_prompt
        self.contextualize_q_prompt = contextualize_q_prompt

    def get_chain(self, retriever):
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, self.contextualize_q_prompt
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain
