from typing import Union, List, Literal
import glob
from tqdm import tqdm
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# def remove_non_utf8_characters(text):
#     return ''.join(char for char in text if ord(char) < 128)
# def remove_non_utf8_characters(text):
#     return text.encode('utf-8', 'ignore').decode('utf-8')


# def load_pdf(pdf_file):
#     docs = PyPDFLoader(pdf_file, extract_images=True).load()
#     for doc in docs:
#         doc.page_content = remove_non_utf8_characters(doc.page_content)
#     return docs

# def load_html(html_file):
#     docs = BSHTMLLoader(html_file,open_encoding='utf-8').load()
#     for doc in docs:
#         doc.page_content = remove_non_utf8_characters(doc.page_content)
#     return docs


# def get_num_cpu():
#     return multiprocessing.cpu_count()


# class BaseLoader:
#     def __init__(self) -> None:
#         self.num_processes = get_num_cpu()

#     def __call__(self, files: List[str], **kwargs):
#         pass


# class PDFLoader(BaseLoader):
#     def __init__(self) -> None:
#         super().__init__()

#     def __call__(self, pdf_files: List[str], **kwargs):
#         num_processes = min(self.num_processes, kwargs["workers"])
#         with multiprocessing.Pool(processes=num_processes) as pool:
#             doc_loaded = []
#             total_files = len(pdf_files)
#             with tqdm(total=total_files, desc="Loading PDFs", unit="file") as pbar:
#                 for result in pool.imap_unordered(load_pdf, pdf_files):
#                     doc_loaded.extend(result)
#                     pbar.update(1)
#         return doc_loaded

# class HTMLLoader(BaseLoader):
#     def __init__(self) -> None:
#         super().__init__()

#     def __call__(self, html_files: List[str], **kwargs):
#         num_processes = min(self.num_processes, kwargs["workers"])
#         with multiprocessing.Pool(processes=num_processes) as pool:
#             doc_loaded = []
#             total_files = len(html_files)
#             with tqdm(total=total_files, desc="Loading HTMLs", unit="file") as pbar:
#                 for result in pool.imap_unordered(load_html, html_files):
#                     doc_loaded.extend(result)
#                     pbar.update(1)
#         return doc_loaded


# class TextSplitter:
#     def __init__(self, 
#                  separators: List[str] = ['\n\n', '\n', ' ', ''],
#                  chunk_size: int = 2048,
#                  chunk_overlap: int = 500,
#                  ) -> None:
        
#         self.splitter = RecursiveCharacterTextSplitter(
#             separators=separators,
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#         )
#     def __call__(self, documents):
#         return self.splitter.split_documents(documents)



# class Loader:
#     def __init__(self, 
#                  file_type: str = Literal["pdf", "html"],
#                  split_kwargs: dict = {
#                      "chunk_size": 2048,
#                      "chunk_overlap": 500}
#                  ) -> None:
#         assert file_type in ["pdf", "html"], "file_type must be either pdf or html"
#         self.file_type = file_type
#         if file_type == "pdf":
#             self.doc_loader = PDFLoader()
#         elif file_type == "html":
#             self.doc_loader = HTMLLoader()
#         else:
#             raise ValueError("file_type must be either pdf or html")

#         self.doc_splitter = TextSplitter(**split_kwargs)

#     def load(self, pdf_files: Union[str, List[str]], workers: int = 1):
#         if isinstance(pdf_files, str):
#             pdf_files = [pdf_files]
#         doc_loaded = self.doc_loader(pdf_files, workers=workers)
#         doc_split = self.doc_splitter(doc_loaded) 

#          # Check the resulting chunks directly after chunking
#         # for i, chunk in enumerate(doc_split):
#         #     print(f"Chunk {i + 1}:\n{chunk.page_content}\n{'=' * 50}\n")

#         return doc_split

#     def load_dir(self, dir_path: str, workers: int = 1):
#         if self.file_type == "pdf":
#             files = glob.glob(f"{dir_path}/*.pdf")
#             assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
#         else:
#             files = glob.glob(f"{dir_path}/*.html")
#             assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
#         return self.load(files, workers=workers)


##############################################
#sematic chunking

from typing import Union, List, Literal
import glob
from tqdm import tqdm
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain.docstore.document import Document
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re


def remove_non_utf8_characters(text):
    return text.encode('utf-8', 'ignore').decode('utf-8')


def load_pdf(pdf_file):
    """Combine all pages of a PDF into a single content block."""
    docs = PyPDFLoader(pdf_file, extract_images=True).load()
    combined_content = " ".join([doc.page_content for doc in docs])
    return [Document(page_content=remove_non_utf8_characters(combined_content), metadata=docs[0].metadata)]


# def load_pdf(pdf_file):
#     docs = PyPDFLoader(pdf_file, extract_images=True).load()
#     return [Document(page_content=remove_non_utf8_characters(doc.page_content), metadata=doc.metadata) for doc in docs]


def load_html(html_file):
    docs = BSHTMLLoader(html_file, open_encoding='utf-8').load()
    return [Document(page_content=remove_non_utf8_characters(doc.page_content), metadata=doc.metadata) for doc in docs]


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


class HTMLLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, html_files: List[str], **kwargs):
        num_processes = min(self.num_processes, kwargs["workers"])
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            total_files = len(html_files)
            with tqdm(total=total_files, desc="Loading HTMLs", unit="file") as pbar:
                for result in pool.imap_unordered(load_html, html_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded

class SemanticChunker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.6, buffer_size: int = 1):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.buffer_size = buffer_size

    def __call__(self, documents: List[Document]):
        chunked_docs = []
        for doc in documents:
            # Step 1: Split text into sentences using regex
            sentences = re.split(r'[.?!]', doc.page_content)
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                continue

            # Step 2: Group adjacent sentences into larger sentences using buffer_size
            grouped_sentences = []
            for i in range(0, len(sentences), self.buffer_size + 1):
                grouped_sentence = ' '.join(sentences[i:i + self.buffer_size + 1])
                grouped_sentences.append(grouped_sentence)

            # Step 3: Encode the sentences into embeddings
            embeddings = self.model.encode(grouped_sentences)
            current_chunk = grouped_sentences[0]
            current_embedding = embeddings[0].reshape(1, -1)

            # Step 4: Calculate similarity scores between consecutive sentences
            for i in range(1, len(grouped_sentences)):
                new_embedding = embeddings[i].reshape(1, -1)
                similarity = cosine_similarity(current_embedding, new_embedding)[0][0]

                # Step 5: Chunk the sentences based on the similarity scores and the threshold
                if similarity < self.threshold:
                    chunked_docs.append(Document(page_content=current_chunk, metadata=doc.metadata))
                    current_chunk = grouped_sentences[i]
                    current_embedding = new_embedding
                else:
                    current_chunk += ' ' + grouped_sentences[i]
                    current_embedding = (current_embedding + new_embedding) / 2

            # Append the last chunk
            if current_chunk:
                chunked_docs.append(Document(page_content=current_chunk, metadata=doc.metadata))

        return chunked_docs


class Loader:
    def __init__(self, 
                 file_type: str = Literal["pdf", "html"],
                 split_kwargs: dict = {
                     "threshold": 0.6}
                 ) -> None:
        assert file_type in ["pdf", "html"], "file_type must be either pdf or html"
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader()
        elif file_type == "html":
            self.doc_loader = HTMLLoader()
        else:
            raise ValueError("file_type must be either pdf or html")

        self.doc_splitter = SemanticChunker(**split_kwargs)

    def load(self, pdf_files: Union[str, List[str]], workers: int = 1):
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        doc_loaded = self.doc_loader(pdf_files, workers=workers)
        doc_split = self.doc_splitter(doc_loaded)

        return doc_split

    def load_dir(self, dir_path: str, workers: int = 1):
        if self.file_type == "pdf":
            files = glob.glob(f"{dir_path}/*.pdf")
            assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
        else:
            files = glob.glob(f"{dir_path}/*.html")
            assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
        return self.load(files, workers=workers)



# from typing import Union, List, Literal
# import glob
# from tqdm import tqdm
# import multiprocessing
# from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
# from langchain.docstore.document import Document
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import re


# def remove_non_utf8_characters(text):
#     return text.encode('utf-8', 'ignore').decode('utf-8')


# def load_pdf_combined_pages(pdf_file):
#     """Combine all pages of a PDF into a single content block."""
#     docs = PyPDFLoader(pdf_file, extract_images=True).load()
#     combined_content = " ".join([doc.page_content for doc in docs])
#     return [Document(page_content=remove_non_utf8_characters(combined_content), metadata=docs[0].metadata)]


# def load_html(html_file):
#     docs = BSHTMLLoader(html_file, open_encoding='utf-8').load()
#     return [Document(page_content=remove_non_utf8_characters(doc.page_content), metadata=doc.metadata) for doc in docs]


# def get_num_cpu():
#     return multiprocessing.cpu_count()


# class BaseLoader:
#     def __init__(self) -> None:
#         self.num_processes = get_num_cpu()

#     def __call__(self, files: List[str], **kwargs):
#         pass


# class PDFLoader(BaseLoader):
#     def __init__(self) -> None:
#         super().__init__()

#     def __call__(self, pdf_files: List[str], **kwargs):
#         num_processes = min(self.num_processes, kwargs["workers"])
#         with multiprocessing.Pool(processes=num_processes) as pool:
#             doc_loaded = []
#             total_files = len(pdf_files)
#             with tqdm(total=total_files, desc="Loading PDFs", unit="file") as pbar:
#                 for result in pool.imap_unordered(load_pdf_combined_pages, pdf_files):
#                     doc_loaded.extend(result)
#                     pbar.update(1)
#         return doc_loaded


# class HTMLLoader(BaseLoader):
#     def __init__(self) -> None:
#         super().__init__()

#     def __call__(self, html_files: List[str], **kwargs):
#         num_processes = min(self.num_processes, kwargs["workers"])
#         with multiprocessing.Pool(processes=num_processes) as pool:
#             doc_loaded = []
#             total_files = len(html_files)
#             with tqdm(total=total_files, desc="Loading HTMLs", unit="file") as pbar:
#                 for result in pool.imap_unordered(load_html, html_files):
#                     doc_loaded.extend(result)
#                     pbar.update(1)
#         return doc_loaded


# class SemanticChunker:
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.6, buffer_size: int = 1):
#         self.model = SentenceTransformer(model_name)
#         self.threshold = threshold
#         self.buffer_size = buffer_size

#     def __call__(self, documents: List[Document]):
#         chunked_docs = []
#         for doc in documents:
#             # Step 1: Split text into sentences using regex
#             content = doc.page_content
#             sentences = re.split(r'[.?!]', content)
#             sentences = [s.strip() for s in sentences if s.strip()]
#             if not sentences:
#                 continue

#             # Step 2: Detect tables and handle them separately
#             table_chunks = re.split(r'(?=Table\s+\d+\.\d+)', content)  # Detect tables using regex
#             for table_chunk in table_chunks:
#                 # Handle regular text and tables differently
#                 if re.match(r'Table\s+\d+\.\d+', table_chunk):
#                     # Keep the entire table as one chunk
#                     chunked_docs.append(Document(page_content=table_chunk.strip(), metadata=doc.metadata))
#                 else:
#                     # Process non-table content into semantic chunks
#                     grouped_sentences = []
#                     for i in range(0, len(sentences), self.buffer_size + 1):
#                         grouped_sentence = ' '.join(sentences[i:i + self.buffer_size + 1])
#                         grouped_sentences.append(grouped_sentence)

#                     # Step 3: Encode sentences into embeddings
#                     embeddings = self.model.encode(grouped_sentences)
#                     current_chunk = grouped_sentences[0]
#                     current_embedding = embeddings[0].reshape(1, -1)

#                     # Step 4: Calculate similarity scores between consecutive sentences
#                     for i in range(1, len(grouped_sentences)):
#                         new_embedding = embeddings[i].reshape(1, -1)
#                         similarity = cosine_similarity(current_embedding, new_embedding)[0][0]

#                         # Step 5: Chunk sentences based on similarity and threshold
#                         if similarity < self.threshold:
#                             chunked_docs.append(Document(page_content=current_chunk, metadata=doc.metadata))
#                             current_chunk = grouped_sentences[i]
#                             current_embedding = new_embedding
#                         else:
#                             current_chunk += ' ' + grouped_sentences[i]
#                             current_embedding = (current_embedding + new_embedding) / 2

#                     # Append the last chunk
#                     if current_chunk:
#                         chunked_docs.append(Document(page_content=current_chunk, metadata=doc.metadata))

#         return chunked_docs



# class Loader:
#     def __init__(self,
#                  file_type: str = Literal["pdf", "html"],
#                  split_kwargs: dict = {
#                      "threshold": 0.6}
#                  ) -> None:
#         assert file_type in ["pdf", "html"], "file_type must be either pdf or html"
#         self.file_type = file_type
#         if file_type == "pdf":
#             self.doc_loader = PDFLoader()
#         elif file_type == "html":
#             self.doc_loader = HTMLLoader()
#         else:
#             raise ValueError("file_type must be either pdf or html")

#         self.doc_splitter = SemanticChunker(**split_kwargs)

#     def load(self, pdf_files: Union[str, List[str]], workers: int = 1):
#         if isinstance(pdf_files, str):
#             pdf_files = [pdf_files]
#         doc_loaded = self.doc_loader(pdf_files, workers=workers)
#         doc_split = self.doc_splitter(doc_loaded)

#         return doc_split

#     def load_dir(self, dir_path: str, workers: int = 1):
#         if self.file_type == "pdf":
#             files = glob.glob(f"{dir_path}/*.pdf")
#             assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
#         else:
#             files = glob.glob(f"{dir_path}/*.html")
#             assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
#         return self.load(files, workers=workers)
