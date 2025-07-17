from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from typing_extensions import TypedDict, List
from langchain_core.documents import Document

llm = init_chat_model(model="gemma3:1b", model_provider="ollama")
embedding = OllamaEmbeddings(model="nomic-embed-text:latest")

vector_db = Chroma(
    collection_name="test",
    embedding_function=embedding
)

loader = ArxivLoader(
    query="Reinforcement Learning",
    load_max_docs=150
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=256)
all_chunks = text_splitter.split_documents(docs)

_ = vector_db.add_documents(all_chunks)

prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant,
    Answer the questions based on the given context **OR ELSE**
    
    Question: {question}
    Context: {context}
    
    Answer:
    """
)

class State(TypeDict):
    question: str
    context: List[Document]
    answer: str

def retreive(state : State):
    retreived_docs = vector_db.similarity_search(state['question'])
    return {'context': retreived_docs}

def generate(state: State):
    pass
 

