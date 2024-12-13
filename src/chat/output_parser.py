from typing import List
import re
from langchain_core.output_parsers import StrOutputParser


# def recursive_extract(text, pattern, default_answer):
#     match = re.search(pattern, text, re.DOTALL)
#     if match:
#         assistant_text = match.group(1).strip()
#         return recursive_extract(assistant_text, pattern, assistant_text)
#     else:
#         return default_answer


# class Str_OutputParser(StrOutputParser):
#     def __init__(self) -> None:
#         super().__init__()
    
#     def parse(self, text: str) -> str:
#         return self.extract_answer(text)
    
    
#     def extract_answer(self,
#                        text_response: str,
#                        patterns: List[str] = [r'\nAssistant:(.*)', r'\nAI:(.*)'],
#                        default = "Sorry, I am not sure how to help with that."
#                        ) -> str:

#         input_text = text_response
#         default_answer = default
#         for pattern in patterns:
#             output_text = recursive_extract(input_text, pattern, default_answer)
#             if output_text != default_answer:
#                 input_text = output_text
#                 default_answer = output_text
        
#         return output_text

def recursive_extract(text, pattern, default_answer):
    match = re.search(pattern, text, re.DOTALL)
    if match:
        assistant_text = match.group(1).strip()
        # Nếu nội dung không thay đổi, dừng đệ quy
        if assistant_text == text:
            return assistant_text
        # Tiếp tục đệ quy nếu nội dung thay đổi
        return recursive_extract(assistant_text, pattern, assistant_text)
    else:
        return default_answer


class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()
    
    def parse(self, text: str) -> str:
        return self.extract_answer(text)
    
    
    def extract_answer(
        self,
        text_response: str,
        patterns: List[str] = [r'Assistant:(.*)', r'AI:(.*)', r'(.*)'],
        default="Sorry, I am not sure how to help with that."
    ) -> str:
        input_text = text_response
        for pattern in patterns:
            output_text = recursive_extract(input_text, pattern, default)
            # Nếu kết quả hợp lệ, trả về ngay
            if output_text != default:
                return output_text
        # Nếu không có mẫu nào khớp, trả về giá trị mặc định
        return default