import os
import json
import time
import requests
import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import BaseOutputParser


class FunctionCallQuizParser(BaseOutputParser):

    def parse(self, text):
        return json.loads(text)

    def parse_result(self, result, *, partial=False):
        return self.parse(
            result[0].message.additional_kwargs["function_call"]["arguments"]
        )



st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("Quiz GPT")


function = {
    "name": "create_quiz",
    "description":"function that takes a list of questions and answers and returns a quiz",
    "parameters":{
        "type":"object",
        "properties": {
            "questions": {
                "type":"array",
                "items":{
                    "type":"object",
                    "properties":{
                        "question":{
                            "type": "string",
                        },
                        "answers":{
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct":{
                                        "type": "boolean"
                                    },
                                },
                                "required": ["answer", "correct"],
                            },

                        },
                    },
                    "required": ["question", "answers"]
                },
            },
        },
        "required":["questions"],
    },
}



llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind(
    function_call={
        "name":"create_quiz"
    },
    functions=[
        function
    ]
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def detect_language(text: str):
    for char in text:
        if "가" <= char <= "힣":
            return "ko"
        if "\u3040" <= char <= "\u30ff":
            return "ja"
        if "\u4e00" <= char <= "\u9fff":
            return "zh"

    return "en"


@st.cache_data(show_spinner=False, ttl=3600)
def search_wikipedia(query: str, limit: int = 3):
    query = query.strip()

    if not query:
        return []

    lang = detect_language(query)
    url = f"https://{lang}.wikipedia.org/w/api.php"

    headers = {
        "User-Agent": "QuizGPT/1.0 (Streamlit Study App)"
    }

    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1,
        "srlimit": limit,
    }

    try:
        search_response = requests.get(
            url,
            params=search_params,
            headers=headers,
            timeout=10,
        )

        search_response.raise_for_status()
        search_data = search_response.json()

        search_results = search_data.get("query", {}).get("search", [])

        if not search_results:
            return []

        docs = []

        for result in search_results:
            title = result.get("title")

            if not title:
                continue

            time.sleep(0.5)

            page_params = {
                "action": "query",
                "prop": "extracts",
                "explaintext": True,
                "exintro": False,
                "titles": title,
                "format": "json",
                "utf8": 1,
            }

            page_response = requests.get(
                url,
                params=page_params,
                headers=headers,
                timeout=10,
            )

            page_response.raise_for_status()
            page_data = page_response.json()

            pages = page_data.get("query", {}).get("pages", {})

            for _, page in pages.items():
                content = page.get("extract", "")

                if content:
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": "wikipedia",
                                "title": title,
                            },
                        )
                    )

        return docs

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            st.error("Wikipedia 요청이 너무 많습니다. 잠시 후 다시 시도하세요.")
        else:
            st.error(f"Wikipedia HTTP 오류가 발생했습니다: {e}")
        return []

    except requests.exceptions.RequestException as e:
        st.error(f"Wikipedia 요청 중 네트워크 오류가 발생했습니다: {e}")
        return []

    except json.JSONDecodeError:
        st.error("Wikipedia 응답을 JSON으로 변환하지 못했습니다.")
        return []


def make_quiz_message(docs):
    context = format_docs(docs)
    language = detect_language(docs[0].page_content) if docs else "en"

    return [
        HumanMessage(
            content=f"""
                Make 10 quiz questions based ONLY on the following context.

                Each question must have 4 answers.
                Only one answer must be correct.
                Create the quiz in this language: {language}

                Context:
                {context}
            """
        )
    ]


questions_chain = (
    RunnableLambda(make_quiz_message)
    | llm
    | FunctionCallQuizParser()
)



@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    os.makedirs("./.cache/quiz_files", exist_ok=True)

    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(docs, source):
    return questions_chain.invoke(docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(_docs, topic):
    return search_wikipedia(topic, limit=3)

if "docs" not in st.session_state:
    st.session_state["docs"] = None


with st.sidebar:
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )

        if file:
            st.session_state["docs"] = split_file(file)

    else:
        topic = st.text_input("Search Wikipedia...")

        if topic:
            current_topic = topic.strip()

            st.session_state["docs"] = wiki_search(
                st.session_state["docs"],
                current_topic
            )
    
    show_answer = st.toggle("Show correct answer after submit")


docs = st.session_state["docs"]


if not docs:
    st.markdown(
        """
Welcome to QuizGPT.

I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

Get started by uploading a file or searching on Wikipedia in the sidebar.
"""
    )

else:
    response = run_quiz_chain(
        docs,
        topic if choice == "Wikipedia Article" and topic else file.name
    )

    with st.form("questions_form"):
        user_answers = []

        for index, question in enumerate(response["questions"]):
            st.write(question["question"])

            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                key=f"question_{index}",
            )

            user_answers.append(
                {
                    "question": question,
                    "selected": value,
                }
            )

        button = st.form_submit_button("Submit")

    if button:
        for item in user_answers:
            question = item["question"]
            selected = item["selected"]

            correct_answer = next(
                answer["answer"]
                for answer in question["answers"]
                if answer["correct"] is True
            )

            st.write(question["question"])

            if selected == correct_answer:
                st.success("Correct!")
            elif selected is None:
                st.warning("No answer selected.")
            else:
                st.error("Wrong!")

                if show_answer:
                    st.info(f"Correct answer: {correct_answer}")