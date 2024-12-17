from typing import Union, List, Literal
import glob, re
from tqdm import tqdm
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from pydantic import BaseModel

def remove_non_utf8_characters(text):
    """Remove non-UTF-8 characters to ensure text integrity."""
    return text.encode('utf-8', 'ignore').decode('utf-8')

def clean_text_advanced(text: str) -> str:
    """Advanced text cleaning, removing unnecessary patterns and fixing formatting."""
    text = remove_non_utf8_characters(text)
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
    return text

# def load_pdf(pdf_file):
#     """Combine all pages of a PDF into a single content block."""
#     docs = PyPDFLoader(pdf_file, extract_images=False).load()
#     combined_content = " ".join([doc.page_content for doc in docs])
#     return [Document(page_content=clean_text_advanced(combined_content), metadata=docs[0].metadata)]


def load_pdf(pdf_file):
    docs = PyPDFLoader(pdf_file, extract_images=False).load()
    return [Document(page_content=clean_text_advanced(doc.page_content), metadata=doc.metadata) for doc in docs]


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


class TextSplitter:
    def __init__(self,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85,
                buffer_size=1,
                sentence_split_regex: str = r"(?<=[.?!])\s+",
                ) -> None:
        
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
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
                 file_type: str = Literal["pdf", "html"],
                #  split_kwargs: dict = {
                #      "threshold": 0.6}
                    split_kwargs: dict = {
                        "breakpoint_threshold_type": "percentile",
                        "breakpoint_threshold_amount": 85,
                        "buffer_size": 1,
                        "sentence_split_regex": r"(?<=[.?!])\s+",
                    }
                 ) -> None:
        assert file_type in ["pdf"], "file_type must be pdf"
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader()
        else:
            raise ValueError("file_type must be pdf")
        
        self.doc_splitter = TextSplitter(**split_kwargs)

    def load(self, pdf_files: Union[str, List[str]], workers: int = 1):
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        doc_loaded = self.doc_loader(pdf_files, workers=workers)
        doc_split = self.doc_splitter(doc_loaded)

        # In số chunk
        print(f"Number of chunks from files: {len(doc_split)}")
        return doc_split

    def load_dir(self, dir_path: str, workers: int = 1):
        if self.file_type == "pdf":
            files = glob.glob(f"{dir_path}/*.pdf")
            assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
        else:
            raise ValueError("file_type must be pdf")
        return self.load(files, workers=workers)
