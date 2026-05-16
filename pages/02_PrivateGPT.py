from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_classic.storage import LocalFileStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.callbacks import BaseCallbackHandler

st.set_page_config(
    page_title="PrivateGPT",
    page_icon="🤖",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

class ChatCallBackHandler(BaseCallbackHandler):

    message =""
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
    
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOllama(
    model="mistral:latest",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallBackHandler()
    ]
)


#   @st.cache_data 데코레이터는
#   if file:
#       retriever = embed_file(file)
#   에서 file에 어떤 파일이 있는지 확인하고 그 안에 파일이 동일하면 ,Streamlit은 
#   def embed_file(file): 함수를 재실행시키지 않는다.
#   그리고 기존에 반환했던 값을 다시 반환한다.
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-5-nano",
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)

    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OllamaEmbeddings(
        model="mistral:latest"
    )

    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )

    vectorstore = FAISS.from_documents(docs, cache_embeddings)

    retriever = vectorstore.as_retriever() 

    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})
    
 

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)
        

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"],message["role"], save=False,)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def format_history(messages):
    return "\n".join(
        f"{message['role']}: {message['message']}"
        for message in messages
    )

prompt = ChatPromptTemplate.from_template(

    """
    Answer the question using ONLY the following context and not your training data.
    If you don't know the answer, just say you don't know.
    DON'T make anything up.

    Context: {context}

    Chat history:
    {history}
    Question: {question}
    """
)

st.title("PrivateGPT")

st.markdown("""
    Welcome!
    
    Use this chatbot to ask questions to an AI about your files!

    Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"])

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file....")
    if message:
        send_message(message, "human")
        history = format_history(st.session_state.get("messages", []))

        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "history": RunnableLambda(lambda _: history),
        } | prompt | llm
        with st.chat_message("ai"):
            chain.invoke(message)
else:
    st.info("Please upload a file.")