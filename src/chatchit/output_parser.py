from typing import List
import re
from langchain_core.output_parsers import StrOutputParser

# Recursively extract the answer from text using a regex pattern
# If the extracted text is the same as the input, stop recursion
# Otherwise, keep extracting from the new text
# If no match, return the default answer
def recursive_extract(text, pattern, default_answer):
    match = re.search(pattern, text, re.DOTALL)
    if match:
        assistant_text = match.group(1).strip()
        # If the content does not change, stop recursion
        if assistant_text == text:
            return assistant_text
        # Continue recursion if the content changes
        return recursive_extract(assistant_text, pattern, assistant_text)
    else:
        return default_answer

# Custom output parser for extracting the assistant's answer from model output
class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        # Initialize the base string output parser
        super().__init__()
    
    def parse(self, text: str) -> str:
        """
        Parse the raw output text and extract the answer using extract_answer.
        """
        return self.extract_answer(text)
    
    def extract_answer(
        self,
        text_response: str,
        patterns: List[str] = [r'Assistant:(.*)', r'AI:(.*)', r'(.*)'],
        default="Sorry, I am not sure how to help with that."
    ) -> str:
        """
        Try to extract the answer from the text_response using a list of regex patterns.
        Returns the first valid match, or a default message if nothing matches.
        """
        input_text = text_response
        for pattern in patterns:
            output_text = recursive_extract(input_text, pattern, default)
            # If a valid result is found, return it immediately
            if output_text != default:
                return output_text
        # If no pattern matches, return the default value
        return default