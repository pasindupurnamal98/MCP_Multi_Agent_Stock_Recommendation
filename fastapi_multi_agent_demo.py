import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor
from langchain_core.messages import convert_to_messages

load_dotenv()

from langchain_core.messages import convert_to_messages


# -----------------------------
# Utility: Run Supervisor Agent
# -----------------------------

async def run_agent(query: str):
    # Initialize MCP client (Bright Data)
    client = MultiServerMCPClient(
        {
            "bright_data": {
                "command": "npx",
                "args": ["@brightdata/mcp"],
                "env": {
                    "API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN"),
                    "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE", "unblocker"),
                    "BROWSER_ZONE": os.getenv("BROWSER_ZONE", "scraping_browser")
                },
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    #model = init_chat_model(model="openai:gpt-4.1", api_key = os.getenv("OPENAI_API_KEY"))

    model = init_chat_model(
        model="azure_openai:gpt-4",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    )

    stock_finder_agent = create_react_agent(model, tools, prompt=""" You are a stock research analyst specializing in the Indian Stock Market (NSE). Your task is to select 2 promising, actively traided NSE-listed stocks for short term trading (buy/sell) based on recent performance, news buzz,volume or technical strength.
    Avoid penny stocks and illiquid companies.
    Output should include stock names, tickers, and brief reasoning for each choice.
    Respond in structured plain text format.""", name = "stock_finder_agent")

    market_data_agent = create_react_agent(model, tools, prompt="""You are a market data analyst for Indian stocks listed on NSE. Given a list of stock tickers (eg RELIANCE, INFY), your task is to gather recent market information for each stock, including:
    - Current price
    - Previous closing price
    - Today's volume
    - 7-day and 30-day price trend
    - Basic Technical indicators (RSI, 50/200-day moving averages)
    - Any notable spkies in volume or volatility
    
    Return your findings in a structured and readable format for each stock, suitable for further analysis by a recommendation engine. Use INR as the currency. Be concise but complete.""", name = "market_data_agent")

    news_alanyst_agent = create_react_agent(model, tools, prompt="""You are a financial news analyst. Given the names or the tickers of Indian NSE listed stocks, your job is to-
    - Search for the most recent news articles (past 3-5 days)
    - Summarize key updates, announcements, and events for each stock
    - Classify each piece of news as positive, negative or neutral
    - Highlist how the news might affect short term stock price
                                            
    Present your response in a clear, structured format - one section per stock.

    Use bullet points where necessary. Keep it short, factual and analysis-oriented""", name = "news_analyst_agent")

    price_recommender_agent = create_react_agent(model, tools, prompt="""You are a trading stratefy advisor for the Indian Stock Market. You are given -
    - Recent market data (current price, volume, trend, indicators)
    - News summaries and sentiment for each stock
        
    Based on this info, for each stock-
    1. Recommend an action : Buy, Sell or Hold
    2. Suggest a specific target price for entry or exit (INR)
    3. Briefly explain the reason behind your recommendation.
        
    Your goal is to provide practical. near-term trading advice for the next trading day.
        
    Keep the response concise and clearly structured.""", name = "price_recommender_agent")


    supervisor = create_supervisor(
        model=model,
        agents=[stock_finder_agent, market_data_agent, news_alanyst_agent, price_recommender_agent],
        prompt=(
            "You are a supervisor managing four agents:\n"
            "- a stock_finder_agent. Assign research-related tasks to this agent and pick 2 promising NSE stocks\n"
            "- a market_data_agent. Assign tasks to fetch current market data (price, volume, trends)\n"
            "- a news_alanyst_agent. Assign task to search and summarize recent news\n"
            "- a price_recommender_agent. Assign task to give buy/sell decision with target price."
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "Do not do any work yourself."
            "Make sure you complete till end and do not ask for proceed in between the task."
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()

    for chunk in supervisor.stream(
    {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        },
    ):
        pretty_print_messages(chunk, last_message=True)

    final_message_history = chunk["supervisor"]["messages"]

if __name__ == "__main__":
    asyncio.run(run_agent("Give me good stock recommendation from NSE"))