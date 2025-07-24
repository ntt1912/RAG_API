import re
from pathlib import Path
from typing import Callable, Union
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from fastapi import HTTPException


# Check if a string is a valid session identifier (alphanumeric, hyphen, underscore)
def _is_valid_identifier(value: str) -> bool:
    valid_characters = re.compile(r"^[a-zA-Z0-9-_]+$")
    return bool(valid_characters.match(value))


# Factory function to create a chat history retriever for a given session
# base_dir: directory to store chat history files
# max_history_length: maximum number of messages to keep in history
def create_session_factory(base_dir: Union[str, Path],
                           max_history_length: int 
                           ) -> Callable[[str], BaseChatMessageHistory]:
    base_dir_ = Path(base_dir) if isinstance(base_dir, str) else base_dir
    # Ensure the base directory exists
    if not base_dir_.exists():
        base_dir_.mkdir(parents=True)

    # Returns a function that retrieves or creates chat history for a session_id
    def get_chat_history(session_id: str) -> FileChatMessageHistory:
        # Validate session_id format
        if not _is_valid_identifier(session_id):
            raise HTTPException(
                status_code=400,
                detail=f"Session ID `{session_id}` is not in a valid format. "
                "Session ID must only contain alphanumeric characters, "
                "hyphens, and underscores.",
            )
        # Path to the session's chat history file
        file_path = base_dir_ / f"{session_id}.json"

        # Load chat history from file
        chat_hist = FileChatMessageHistory(str(file_path))
        messages = chat_hist.messages

        # If history is too long, keep only the most recent messages
        if len(messages) > max_history_length:
            chat_hist.clear()
            for message in messages[-max_history_length:]:
                chat_hist.add_message(message)

        print("chat_hist len: ", len(chat_hist.messages))
        return chat_hist

    return get_chat_history
