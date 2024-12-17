import re
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
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

qa_system_prompt = """Sử dụng thông tin sau đây để trả lời câu hỏi, hãy trả lời vào đúng trọng tâm của câu hỏi.
Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.
{context}
"""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)


class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()
    
    def parse(self, text: str) -> str:
        return self.extract_answer(text)
    
    def extract_answer(self, text_response: str, pattern: str = r"Answer:\s*(.*)") -> str:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response


class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = rag_prompt
        self.contextualize_q_prompt = contextualize_q_prompt
        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever):
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, self.contextualize_q_prompt
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain

    def format_docs(self, docs):
        context = "\n\n".join(doc.page_content for doc in docs)
        return context