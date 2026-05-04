from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
def create_rag_pipeline(pdf_path):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)
    docs = splitter.split_documents(documents)

    # Create embeddings
    embeddings = OllamaEmbeddings(model="llama2")

    # Store in FAISS
    db = FAISS.from_documents(docs, embeddings)

    # LLM (local model)
    llm = Ollama(model="phi")

    return db, llm


def ask_question(db, llm, query):
    retriever = db.as_retriever(search_kwargs={"k": 3})

    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    return response
