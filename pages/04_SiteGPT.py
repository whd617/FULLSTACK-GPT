from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from difflib import SequenceMatcher
import streamlit as st

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "question_cache" not in st.session_state:
    st.session_state["question_cache"] = {}

if "current_question" not in st.session_state:
    st.session_state["current_question"] = None

if "question_memory" not in st.session_state:
    st.session_state["question_memory"] = []

if "history_limit" not in st.session_state:
    st.session_state["history_limit"] = 6

class ChatCallBackHandler(BaseCallbackHandler):

    def __init__(self):
        self.message = ""
        self.message_box = None
    
    def on_llm_start(self, *args, **kwargs):
        self.message = ""

        with st.chat_message("ai"):
            self.message_box = st.empty()

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token.replace("$", "\$")
        self.message_box.markdown(self.message)

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

        if st.session_state.get("current_question"):
            st.session_state["question_cache"][st.session_state["current_question"]] = self.message

llm_config = {
    "model": "gpt-5-nano",
    "temperature": 0.1,
}

llm = ChatOpenAI(**llm_config)

streaming_llm = ChatOpenAI(
    **llm_config,
    streaming=True,
    callbacks=[ChatCallBackHandler()]
)



answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    history = inputs["history"]

    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "history": history,
        "answers":[
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"]
            }
            for doc in docs
        ]
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Also consider the chat history to understand follow-up questions.

            Chat history:
            {history}

            Answers:
            {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    history = inputs["history"]

    condensed = "\n\n".join(
        f"{answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}\n"
        for answer in answers
    )

    choose_chain = choose_prompt | streaming_llm

    return choose_chain.invoke({
        "question": question,
        "answers": condensed,
        "history": history,
    })

def format_history(messages, limit=6):
    recent_messages = messages[-limit:]

    return "\n\n".join(
        f"{message['role']}: {message['message']}"
        for message in recent_messages
    )

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    
    text = soup.get_text()

    for i in range(1, 100):
        text = text.replace(f"Page {i}", "")

    text = text.replace("...", "").replace("Learn more", "")
    return text

@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )

    loader = SitemapLoader(url,
                           parsing_function=parse_page
                           )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message,role)

def paint_message_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})

def find_similar_cached_answer(query, url, threshold=0.75):
    memories = st.session_state.get("question_memory", [])

    best_score = 0
    best_answer = None

    for item in memories:
        if item["url"] != url:
            continue

        score = SequenceMatcher(
            None,
            query.lower().strip(),
            item["question"].lower().strip()
        ).ratio()

        if score > best_score:
            best_score = score
            best_answer = item["answer"]

    if best_score >= threshold:
        return best_answer

    return None

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.title("Quiz GPT")



st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://example.com")

send_message("I'm ready! Search your Website", "ai", save=False)

if url:
    paint_message_history()
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL")
    else:
        retriever = load_website(url)
        query = st.chat_input("Ask a question to the website.")
        if query:
            send_message(query, "human")

            cache_key = f"{url}:{query}"

            if cache_key in st.session_state["question_cache"]:
                cached_answer = st.session_state["question_cache"][cache_key]
                send_message(cached_answer, "ai")

            else:
                similar_answer = find_similar_cached_answer(query, url)

                if similar_answer:
                    send_message(similar_answer, "ai")

                else:
                    st.session_state["current_question"] = cache_key

                    history = format_history(
                        st.session_state["messages"],
                        st.session_state["history_limit"]
                    )

                    chain = {
                        "docs": retriever,
                        "question": RunnablePassthrough(),
                        "history": RunnableLambda(lambda _: history),
                    } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)

                    chain.invoke(query)

                    if cache_key in st.session_state["question_cache"]:
                        st.session_state["question_memory"].append({
                            "url": url,
                            "question": query,
                            "answer": st.session_state["question_cache"][cache_key],
                        })

                    st.session_state["current_question"] = None
else:
    send_message("Please upload a Url", "ai", save=False)