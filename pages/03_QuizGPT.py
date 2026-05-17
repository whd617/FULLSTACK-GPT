import os
import json
import time
import requests
import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser


class JsonOutputParser(BaseOutputParser):

    def parse(self, text):

        text = text.strip()

        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()

        elif text.startswith("```"):
            text = text.replace("```", "").strip()

        return json.loads(text)


output_parser = JsonOutputParser()


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("Quiz GPT")


llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
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


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant that is role playing as a teacher.

Based ONLY on the following context, make 10 questions to test the user's knowledge about the text.

Each question should have 4 answers.
Three answers must be incorrect and one answer must be correct.

Use (o) to signal the correct answer.

Question examples:

Question: What is the color of the ocean?
Answers: Red|Yellow|Green|Blue(o)

Question: What is the capital of Georgia?
Answers: Baku|Tbilisi(o)|Manila|Beirut

Question: When was Avatar released?
Answers: 2007|2001|2009(o)|1998

Question: Who was Julius Caesar?
Answers: A Roman Emperor(o)|Painter|Actor|Model

Your turn!

Context: {context}

Language: {language}

Create the quiz in the same language as the language above.
""",
        )
    ]
)


questions_chain = (
    {
        "context": format_docs,
        "language": lambda docs: detect_language(
            docs[0].page_content
        ) if docs else "en",
    }
    | questions_prompt
    | llm
    | StrOutputParser()
)


formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a powerful formatting algorithm.

You format exam questions into valid JSON format.
Answers with (o) are the correct ones.

Return ONLY valid JSON.
Do not use markdown.
Do not wrap the result in ```json.

The JSON format must be:

{{
  "questions": [
    {{
      "question": "Question text",
      "answers": [
        {{
          "answer": "Answer text",
          "correct": false
        }},
        {{
          "answer": "Answer text",
          "correct": true
        }},
        {{
          "answer": "Answer text",
          "correct": false
        }},
        {{
          "answer": "Answer text",
          "correct": false
        }}
      ]
    }}
  ]
}}

Questions: {context}
""",
        )
    ]
)


formatting_chain = (
    formatting_prompt
    | llm
    | StrOutputParser()
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
    chain = (
        {"context": questions_chain}
        | formatting_chain
        | output_parser
    )
    return chain.invoke(docs)


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
    start = st.button("Generate Quiz")

    if start:
        response = run_quiz_chain(
            docs,
            topic if topic else file.name
        )

        st.write(response)