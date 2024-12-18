from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.chat.history import create_session_factory
from src.chat.output_parser import Str_OutputParser
from enum import Enum



chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all questions to the best of your ability"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{human_input}"),
    ]
)

class ModelName(str, Enum):
    GPT3_5 = "gpt3.5-turbo"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"

class InputChat(BaseModel):
    human_input: str = Field(
        ...,
        description="The human input to the chat system.",
        extra={"widget": {"type": "chat", "input": "human_input"}},
    )
    session_id: str = Field(
        default=None,
        title="Optional session ID. If not provided, one will be generated.",
    )
    model: ModelName = Field(
        default=ModelName.GEMINI_1_5_FLASH,
        title="Model to use for answering the question",
    )
    model_config = {
        'protected_namespaces': ()
    }

class OutputChat(BaseModel):
    answer: str = Field(..., title="Answer from the model")
    session_id: str = Field(..., title="Session ID for the conversation")
    model: ModelName
    model_config = {"protected_namespaces": ()}


def build_chat_chain(llm, history_folder, max_history_length):

    chain = chat_prompt | llm | Str_OutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        create_session_factory(base_dir=history_folder, 
                               max_history_length=max_history_length),
        input_messages_key="human_input",
        history_messages_key="chat_history",
    )
    return chain_with_history.with_types(input_type=InputChat)
