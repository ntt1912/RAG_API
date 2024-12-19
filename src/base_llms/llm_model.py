# import torch
# from transformers import BitsAndBytesConfig
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain_huggingface import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


# nf4_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# def get_hf_llm(model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", 
#             max_new_token = 1024, 
#             **kwargs):
    
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=nf4_config,
#         low_cpu_mem_usage=True
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=True)

#     model_pipeline = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=max_new_token,
#         pad_token_id=tokenizer.eos_token_id,
#         device_map="auto"
#     )

#     llm = HuggingFacePipeline(
#         pipeline=model_pipeline,
#         model_kwargs=kwargs
#     )

#     return llm



def get_llm(api_key, model_name="gemini-1.5-flash"):
    if not api_key:
        raise ValueError("API key is missing. Please set your API key.")
    
    if model_name == "gpt-4o-mini":
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.2
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=api_key,
            temperature=0.1
        )
    return llm