from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor
from langchain_core.messages import convert_to_messages
from typing import List, Dict, Any

load_dotenv()

app = FastAPI(title="Stock Analysis API", version="1.0.0")

# Request/Response models
class StockAnalysisRequest(BaseModel):
    query: str = ""

class StockAnalysisResponse(BaseModel):
    status: str
    data: List[Dict[str, Any]]
    final_message: str

class HealthResponse(BaseModel):
    status: str
    message: str

# Global variables to store agents (initialized once)
supervisor = None

async def initialize_agents():
    """Initialize all agents and supervisor"""
    global supervisor
    
    if supervisor is not None:
        return supervisor
    
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
    
    model = init_chat_model(
        model="azure_openai:gpt-4",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    )

    # Create agents (you can add your prompts here)
    stock_finder_agent = create_react_agent(
        model, tools, 
        prompt=""" You are a stock research analyst specializing in the Indian Stock Market (NSE). 
                Your task is to select 2 promising, actively traided NSE-listed stocks for short term trading (buy/sell) 
                based on recent performance, news buzz,volume or technical strength.
                Avoid penny stocks and illiquid companies.
                Output should include stock names, tickers, and brief reasoning for each choice.
                Respond in structured plain text format.""", 
        name="stock_finder_agent"
    )

    market_data_agent = create_react_agent(
        model, tools, 
        prompt="""You are a market data analyst for Indian stocks listed on NSE. Given a list of stock tickers (eg RELIANCE, INFY), 
        your task is to gather recent market information for each stock, including:
        - Current price
        - Previous closing price
        - Today's volume
        - 7-day and 30-day price trend
        - Basic Technical indicators (RSI, 50/200-day moving averages)
        - Any notable spkies in volume or volatility
    
        Return your findings in a structured and readable format for each stock, suitable for further analysis by a recommendation engine. 
        Use INR as the currency. Be concise but complete.""", 

        name="market_data_agent"
    )

    news_analyst_agent = create_react_agent(
        model, tools, 
        prompt="""You are a financial news analyst. Given the names or the tickers of Indian NSE listed stocks, your job is to-
        - Search for the most recent news articles (past 3-5 days)
        - Summarize key updates, announcements, and events for each stock
        - Classify each piece of news as positive, negative or neutral
        - Highlist how the news might affect short term stock price
                                                
        Present your response in a clear, structured format - one section per stock.

        Use bullet points where necessary. Keep it short, factual and analysis-oriented""", 
        name="news_analyst_agent"
    )

    price_recommender_agent = create_react_agent(
        model, tools, 
        prompt="""You are a trading stratefy advisor for the Indian Stock Market. You are given -
            - Recent market data (current price, volume, trend, indicators)
            - News summaries and sentiment for each stock
                
            Based on this info, for each stock-
            1. Recommend an action : Buy, Sell or Hold
            2. Suggest a specific target price for entry or exit (INR)
            3. Briefly explain the reason behind your recommendation.
                
            Your goal is to provide practical. near-term trading advice for the next trading day.
                
            Keep the response concise and clearly structured.""", 
        name="price_recommender_agent"
    )

    supervisor = create_supervisor(
        model=model,
        agents=[stock_finder_agent, market_data_agent, news_analyst_agent, price_recommender_agent],
        prompt="You are a supervisor managing four agents:\n"
            "- a stock_finder_agent. Assign research-related tasks to this agent and pick 2 promising NSE stocks\n"
            "- a market_data_agent. Assign tasks to fetch current market data (price, volume, trends)\n"
            "- a news_alanyst_agent. Assign task to search and summarize recent news\n"
            "- a price_recommender_agent. Assign task to give buy/sell decision with target price."
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "Do not do any work yourself."
            "Make sure you complete till end and do not ask for proceed in between the task.",
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()
    
    return supervisor

def process_messages(messages):
    """Process and format messages for API response"""
    processed_data = []
    
    for message in messages:
        if hasattr(message, 'content'):
            processed_data.append({
                "role": getattr(message, 'role', 'assistant'),
                "content": message.content,
                "type": getattr(message, 'type', 'text')
            })
    
    return processed_data

@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup"""
    try:
        await initialize_agents()
        print("Agents initialized successfully")
    except Exception as e:
        print(f"Failed to initialize agents: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", message="Stock Analysis API is running")

@app.post("/analyze-stocks", response_model=StockAnalysisResponse)
async def analyze_stocks(request: StockAnalysisRequest):
    """Main endpoint for stock analysis"""
    try:
        # Initialize agents if not already done
        supervisor_agent = await initialize_agents()
        
        if supervisor_agent is None:
            raise HTTPException(status_code=500, detail="Failed to initialize agents")
        
        # Collect all messages
        all_messages = []
        final_chunk = None
        
        # Stream through supervisor
        async for chunk in supervisor_agent.astream({
            "messages": [
                {
                    "role": "user",
                    "content": request.query or "Analyze promising NSE stocks for short-term trading",
                }
            ]
        }):
            final_chunk = chunk
            # You can process intermediate chunks here if needed
        
        # Process final messages
        if final_chunk and "supervisor" in final_chunk:
            final_messages = final_chunk["supervisor"]["messages"]
            processed_data = process_messages(final_messages)
            
            # Get the last message as final result
            final_message = ""
            if final_messages:
                last_message = final_messages[-1]
                final_message = getattr(last_message, 'content', 'Analysis completed')
            
            return StockAnalysisResponse(
                status="success",
                data=processed_data,
                final_message=final_message
            )
        else:
            raise HTTPException(status_code=500, detail="No response from supervisor")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Stock Analysis API", "docs": "/docs"}

# Additional endpoints you might want
@app.post("/quick-analysis")
async def quick_analysis():
    """Quick analysis with default query"""
    return await analyze_stocks(StockAnalysisRequest(query=""))

@app.get("/agents/status")
async def agents_status():
    """Check if agents are initialized"""
    global supervisor
    return {
        "agents_initialized": supervisor is not None,
        "available_endpoints": ["/analyze-stocks", "/quick-analysis", "/health"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)