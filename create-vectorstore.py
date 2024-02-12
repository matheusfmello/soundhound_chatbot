from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
import re

load_dotenv()

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=20,  # Overlap between chunks
    tokens_per_chunk=200  # Number of tokens per chunk
)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

files = ['soundhound-general.txt', 'soundhound-products.txt', 'soundhound-services.txt']

for file in files:
    
    loader = TextLoader(file, encoding='utf-8')
    documents = loader.load()

    docs = token_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embeddings)
    
    short_name = re.search('-(.*?)\.', file).group(1) # takes whatever is between '-' and '.' from the file name

    db.save_local(f"faiss_{short_name}")