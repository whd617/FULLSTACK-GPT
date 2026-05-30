from typing import Type
import requests
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.globals import set_debug
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import SystemMessage
import streamlit as st

st.set_page_config(
    page_title="InvestorGPT",
    page_icon="🧰"
)

st.markdown(
    """
        # InvestorGPT

        Welcome to InvestorGPT.

        Write down the name of a company and our Agent will do the research for you.
    """
)



# verbose=True 대체
set_debug(True)

alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")

llm = ChatOpenAI(
    model= "gpt-5-nano",
    temperature=0.1,
)


class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")

class StockMarketSymbolSearchTool(BaseTool):
    name: str = "StockMarketSymbolSearchTool"
    description: str = """
    Use this tool to find the stock market symbol for a company.
    It takes a query as an argument.
    Example query: Stock Market Symbol for Apple Company.
    """
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = StockMarketSymbolSearchToolArgsSchema

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)

class CompanyOverviewArgsSchema(BaseModel):
    symbol: str = Field(description="Stock symbol of the company. Example: AAPL, TSLA")

class CompanyOverviewTool(BaseTool):
    name: str = "CompanyOverview"
    description: str = """
    Use this to get an overview of the financials of the company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}")
        return r.json()

class CompanyIncomeStatementTool(BaseTool):
    name: str = "CompanyIncomeStatement"
    description: str = """
    Use this to get the income statement of a company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}")
        return r.json()

class CompanyStockPerformanceTool(BaseTool):
    name: str = "CompanyStockPerformanceTool"
    description: str = """
    Use this to get the weekly performance of a company stock.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={alpha_vantage_api_key}")
        response =r.json()
        return list(response["Time Series (Daily)"].items())[:200]

agent = create_agent(
    model=llm,
        system_prompt=SystemMessage(content="""
            You are a helpful assistant.
            Use tools when needed.

            You are a hedge fund manager.

            You evaluate a company and provide your opinion and reasons why the stock is a buy or not.
                                    
            Consider the performance of a stock, the company overview and the income statement.
                                    
            Be assertive in your judgement and recommend the stock or advise the user against it.
            """
        ),
    tools=[
        StockMarketSymbolSearchTool(),
        CompanyOverviewTool(),
        CompanyIncomeStatementTool(),
        CompanyStockPerformanceTool()
    ],
)


prompt = """
Give me financial information on Clouldflare's stock, 
considering its financials, income statements and stock performance help me analyze if it's a potential good investment.
"""


result = agent.invoke({
    "messages": [
        {"role": "user", "content": prompt}
    ]
})

print(result["messages"])

company = st.text_input("Write the name of the company you are interested on.")

if company:
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": company}
        ]
    })

    st.write(result["messages"][-1].content)