from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

load_dotenv()

pdf_path = Path(__file__).parent.parent / "BeginningNodejs.pdf"


pdfLoader = PyPDFLoader(file_path=str(pdf_path))

pages = pdfLoader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 400)

chunks = text_splitter.split_documents(documents = pages)

#Vector embeddings
embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")

#bridge for qdrant db
vectorStore = QdrantVectorStore.from_documents(
    documents = chunks,
    embedding = embeddings,
    collection_name = "beginning-nodejs",
    url = "http://localhost:6333")