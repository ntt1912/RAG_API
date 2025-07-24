# Standard library imports
from typing import List
import re, os
import multiprocessing

# Third-party imports
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

def remove_non_utf8_characters(text):
    """
    Remove non-UTF-8 characters from text to ensure text integrity.
    """
    return text.encode('utf-8', 'ignore').decode('utf-8')

def clean_text_advanced(text: str) -> str:
    """
    Clean and normalize text by removing non-UTF-8 characters and extra whitespace.
    """
    text = remove_non_utf8_characters(text)
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
    return text

def load_pdf(pdf_file):
    """
    Load a PDF file and combine all pages into a single Document object.
    Returns a list with one Document containing all content and metadata from the first page.
    """
    docs = PyPDFLoader(pdf_file, extract_images=False).load()
    combined_content = " ".join([doc.page_content for doc in docs])
    return [Document(page_content=clean_text_advanced(combined_content), metadata=docs[0].metadata)]

def load_docx(docx_file):
    """
    Load a DOCX file and return a list of Document objects, one per document part.
    Each Document contains cleaned text and its metadata.
    """
    docs = Docx2txtLoader(docx_file).load()
    return [Document(page_content=clean_text_advanced(doc.page_content), metadata=doc.metadata) for doc in docs]

# Alternative PDF loader (not used):
# def load_pdf(pdf_file):
#     docs = PyPDFLoader(pdf_file, extract_images=False).load()
#     return [Document(page_content=clean_text_advanced(doc.page_content), metadata=doc.metadata) for doc in docs]

def get_num_cpu():
    """
    Get the number of available CPU cores for multiprocessing.
    """
    return multiprocessing.cpu_count()

class BaseLoader:
    """
    Abstract base loader class for document loaders using multiprocessing.
    """
    def __init__(self) -> None:
        self.num_processes = get_num_cpu()

    def __call__(self, files: List[str], **kwargs):
        # To be implemented by subclasses
        pass

class PDFLoader(BaseLoader):
    """
    Loader for processing multiple PDF files in parallel using multiprocessing.
    """
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pdf_files: List[str], **kwargs):
        # Determine number of processes to use
        num_processes = min(self.num_processes, kwargs["workers"])
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            total_files = len(pdf_files)
            # Progress bar for loading PDFs
            with tqdm(total=total_files, desc="Loading PDFs", unit="file") as pbar:
                for result in pool.imap_unordered(load_pdf, pdf_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded

class DOCXLoader(BaseLoader):
    """
    Loader for processing multiple DOCX files in parallel using multiprocessing.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, docx_files: List[str], **kwargs):
        # Determine number of processes to use
        num_processes = min(self.num_processes, kwargs["workers"])
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            total_files = len(docx_files)
            # Progress bar for loading DOCXs
            with tqdm(total=total_files, desc="Loading DOCXs", unit="file") as pbar:
                for result in pool.imap_unordered(load_docx, docx_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded

class TextSplitter:
    """
    Splits documents into semantically meaningful chunks using embeddings and a semantic chunker.
    """
    def __init__(self,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85,
                buffer_size=1,
                sentence_split_regex: str = r"(?<=[.?!])\s+",
                ) -> None:
        # Use OpenAI embeddings for semantic chunking
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1024, api_key=openai_api_key)
        self.splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            buffer_size=buffer_size,
            sentence_split_regex=sentence_split_regex,
        )

    def __call__(self, documents):
        """
        Split a list of Document objects into semantic chunks.
        """
        return self.splitter.split_documents(documents)

class Loader:
    """
    High-level loader that loads and splits files of supported types (PDF, DOCX).
    Handles batching, parallel loading, and semantic splitting.
    """
    def __init__(self, 
                 file_types: List[str] = ["pdf", "docx"],
                 split_kwargs: dict = {
                     "breakpoint_threshold_type": "percentile",
                     "breakpoint_threshold_amount": 85,
                     "buffer_size": 1,
                     "sentence_split_regex": r"(?<=[.?!])\s+",
                 }
                 ) -> None:
        # Ensure only supported file types are used
        assert all(ft in ["pdf", "docx"] for ft in file_types), \
            "file_types must only contain 'pdf'or 'docx'"
        self.file_types = file_types
        self.doc_splitter = TextSplitter(**split_kwargs)
        self.loaders = {
            "pdf": PDFLoader(),
            "docx": DOCXLoader(),
        }

    def load_and_split(self, files: List[str], workers: int = 4):
        """
        Load files of supported types, split them into semantic chunks, and return the chunks.
        """
        doc_loaded = []
        for file_type in self.file_types:
            specific_files = [file for file in files if file.endswith(f".{file_type}")]
            if specific_files:
                doc_loaded.extend(self.loaders[file_type](specific_files, workers=workers))
        doc_split = self.doc_splitter(doc_loaded)

        # Print the number of chunks created
        print(f"Number of chunks from files: {len(doc_split)}")
        return doc_split

    def load_dir(self, file_path: str, workers: int = 4):
        """
        Load and split a single file (by path) using the default number of workers.
        """
        return self.load_and_split([file_path], workers=workers)



