from typing import List
import re, os
from tqdm import tqdm
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

def remove_non_utf8_characters(text):
    """Remove non-UTF-8 characters to ensure text integrity."""
    return text.encode('utf-8', 'ignore').decode('utf-8')

def clean_text_advanced(text: str) -> str:
    """Advanced text cleaning, removing unnecessary patterns and fixing formatting."""
    text = remove_non_utf8_characters(text)
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
    return text

def load_pdf(pdf_file):
    """Combine all pages of a PDF into a single content block."""
    docs = PyPDFLoader(pdf_file, extract_images=False).load()
    combined_content = " ".join([doc.page_content for doc in docs])
    return [Document(page_content=clean_text_advanced(combined_content), metadata=docs[0].metadata)]


def load_docx(docx_file):
    """Load a DOCX file and return a list of documents."""
    docs = Docx2txtLoader(docx_file).load()
    return [Document(page_content=clean_text_advanced(doc.page_content), metadata=doc.metadata) for doc in docs]

# def load_pdf(pdf_file):
#     docs = PyPDFLoader(pdf_file, extract_images=False).load()
#     return [Document(page_content=clean_text_advanced(doc.page_content), metadata=doc.metadata) for doc in docs]


def get_num_cpu():
    return multiprocessing.cpu_count()


class BaseLoader:
    def __init__(self) -> None:
        self.num_processes = get_num_cpu()

    def __call__(self, files: List[str], **kwargs):
        pass


class PDFLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pdf_files: List[str], **kwargs):
        num_processes = min(self.num_processes, kwargs["workers"])
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            total_files = len(pdf_files)
            with tqdm(total=total_files, desc="Loading PDFs", unit="file") as pbar:
                for result in pool.imap_unordered(load_pdf, pdf_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded

class DOCXLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, docx_files: List[str], **kwargs):
        num_processes = min(self.num_processes, kwargs["workers"])
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            total_files = len(docx_files)
            with tqdm(total=total_files, desc="Loading DOCXs", unit="file") as pbar:
                for result in pool.imap_unordered(load_docx, docx_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded

class TextSplitter:
    def __init__(self,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85,
                buffer_size=1,
                sentence_split_regex: str = r"(?<=[.?!])\s+",
                ) -> None:
        
        self.embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small', dimensions = 1024,api_key=openai_api_key)
        self.splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            buffer_size=buffer_size,
            sentence_split_regex=sentence_split_regex,
        )
    def __call__(self, documents):
        return self.splitter.split_documents(documents)


class Loader:
    def __init__(self, 
                 file_types: List[str] = ["pdf", "docx"],
                 split_kwargs: dict = {
                     "breakpoint_threshold_type": "percentile",
                     "breakpoint_threshold_amount": 85,
                     "buffer_size": 1,
                     "sentence_split_regex": r"(?<=[.?!])\s+",
                 }
                 ) -> None:
        assert all(ft in ["pdf", "docx"] for ft in file_types), \
            "file_types must only contain 'pdf'or 'docx'"
        self.file_types = file_types
        self.doc_splitter = TextSplitter(**split_kwargs)
        self.loaders = {
            "pdf": PDFLoader(),
            "docx": DOCXLoader(),
        }

    def load_and_split(self, files: List[str], workers: int = 1):
        doc_loaded = []
        for file_type in self.file_types:
            specific_files = [file for file in files if file.endswith(f".{file_type}")]
            if specific_files:
                doc_loaded.extend(self.loaders[file_type](specific_files, workers=workers))
        doc_split = self.doc_splitter(doc_loaded)

        # In số chunk
        print(f"Number of chunks from files: {len(doc_split)}")
        return doc_split

    def load_dir(self, file_path: str, workers: int = 1):
        return self.load_and_split([file_path], workers=workers)



