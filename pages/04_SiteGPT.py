from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
import streamlit as st

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.title("Quiz GPT")

# 웹사이트의 html 코드를 text로 편환하는 역할
html2text_transformer = Html2TextTransformer()

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://example.com")

if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(transformed)