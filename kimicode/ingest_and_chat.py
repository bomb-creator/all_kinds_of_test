# =================== ingest_and_chat.py ===================
"""
One-shot script:
  1) Upload a PDF to your Qdrant Cloud collection
  2) Chat with it via Gemini
"""

# ---------------------------------------------------------
# 0.  CONFIG – EDIT ONLY THESE TWO LINES
# ---------------------------------------------------------
GOOGLE_API_KEY   = "AIzaSyAHpD3cFPogSSoLjWZmi7cQ3pHSlDyBANU"
PDF_PATH         = "/workspaces/all_kinds_of_test/my_document.pdf"          # any local PDF

# ---------------------------------------------------------
# 1.  Auto-install deps
# ---------------------------------------------------------
import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "langchain", "langchain-google-genai", "langchain-community",
    "qdrant-client", "pypdf", "python-dotenv"
])

# ---------------------------------------------------------
# 2.  Imports
# ---------------------------------------------------------
import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL")
PDF_PATH       = "/workspaces/all_kinds_of_test/my_document.pdf"  # any local PDF

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient

# ---------------------------------------------------------
# 3.  Connect to YOUR Qdrant Cloud cluster
# ---------------------------------------------------------
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

print("✅ Connected to Qdrant:", qdrant_client.get_collections())

# ---------------------------------------------------------
# 4.  Build Gemini components
# ---------------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm        = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

# ---------------------------------------------------------
# 5.  Load & chunk the PDF
# ---------------------------------------------------------
loader      = PyPDFLoader(PDF_PATH)
raw_docs    = loader.load()
splitter    = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks      = splitter.split_documents(raw_docs)
print(f"📄 Loaded {len(chunks)} chunks")

# ---------------------------------------------------------
# 6.  Upsert to Qdrant Cloud
#    DO NOT pass a pre-built client; use url + api_key
# ---------------------------------------------------------
COLLECTION_NAME = "pdf_docs"
vectorstore = Qdrant.from_documents(
    documents      = chunks,
    embedding      = embeddings,
    url            = QDRANT_URL,
    api_key        = QDRANT_API_KEY,
    collection_name= COLLECTION_NAME,
    batch_size     = 64
)

# ---------------------------------------------------------
# 7.  Retrieval QA chain
# ---------------------------------------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Use only the provided context to answer the question.\n\n{context}"),
    ("human", "{question}")
])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)

# ---------------------------------------------------------
# 8.  Interactive chat loop
# ---------------------------------------------------------
print("\n🤖 Ready! Ask questions about the PDF (type 'exit' to quit).\n")
while True:
    q = input("> ")
    if q.strip().lower() in {"exit", "quit"}:
        break
    answer = qa_chain.invoke(q)["result"]
    print(answer, "\n")