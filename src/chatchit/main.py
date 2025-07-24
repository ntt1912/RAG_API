from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.chatchit.history import create_session_factory
from src.chatchit.output_parser import Str_OutputParser
from enum import Enum

# Define the chat prompt template for the assistant
# Includes a system message, chat history, and the latest human input
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all questions to the best of your ability"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{human_input}"),
    ]
)

# Enum for supported model names
class ModelName(str, Enum):
    GPT3_5 = "gpt3.5-turbo"              # OpenAI GPT-3.5 Turbo
    GEMINI_1_5_FLASH = "gemini-1.5-flash"  # Google Gemini 1.5 Flash

# Input schema for a chat request
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
    # Pydantic config for advanced options (not used here)
    model_config = {
        'protected_namespaces': ()
    }

# Output schema for a chat response
class OutputChat(BaseModel):
    answer: str = Field(..., title="Answer from the model")  # Model's answer
    session_id: str = Field(..., title="Session ID for the conversation")  # Session tracking
    model: ModelName  # Model used for the answer
    model_config = {"protected_namespaces": ()}

# Build the chat chain with message history and output parsing
# llm: language model instance
# history_folder: directory to store chat histories
# max_history_length: maximum number of messages to keep in history
def build_chat_chain(llm, history_folder, max_history_length):
    # Compose the chain: prompt -> LLM -> output parser
    chain = chat_prompt | llm | Str_OutputParser()
    # Add message history management
    chain_with_history = RunnableWithMessageHistory(
        chain,
        create_session_factory(base_dir=history_folder, 
                               max_history_length=max_history_length),
        input_messages_key="human_input",
        history_messages_key="chat_history",
    )
    # Return the chain with input type enforcement
    return chain_with_history.with_types(input_type=InputChat)
