from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import ArxivLoader, TextLoader
from langchain_community.document_loaders.pdf import BasePDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START


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

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retreive(state : State):
    retreived_docs = vector_db.similarity_search(state['question'])
    return {'context': retreived_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    message = prompt.invoke(
        {"question":state['question'], "context":docs_content}
    )
    response = llm.invoke(message)
    return {"answer":response.content}

graph_builder = StateGraph(State).add_sequence([retreive, generate])
graph_builder.add_edge(START, "retreive")
graph = graph_builder.compile()

response = graph.invoke({"question": "What is RL? be very thorrough"})
print(response['answer'])

