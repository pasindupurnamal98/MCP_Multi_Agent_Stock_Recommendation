import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

load_dotenv()

async def run_agent():
    client = MultiServerMCPClient(
        {
            "bright_data": {
                "command": "npx",
                "args": ["@brightdata/mcp"],
                "env": {
                    "API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN"),
                    "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE", "unblocker"),
                    "BROWSER_ZONE": os.getenv("BROWSER_ZONE", "scraping_browser"),
                },
                "transport": "stdio",
            },
        }
    )

    tools = await client.get_tools()

    model = init_chat_model(
        model="azure_openai:gpt-4",   # important: use azure_openai prefix
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    agent = create_react_agent(
        model,
        tools,
        prompt="You are a web search agent with access to Bright Data tools to fetch real-time flight information."
    )

    user_query = "Tell me Current Flights from New York to San Francisco"

    agent_response = await agent.ainvoke({
        "messages": [{"role": "user", "content": user_query}]
    })

    print(agent_response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(run_agent())