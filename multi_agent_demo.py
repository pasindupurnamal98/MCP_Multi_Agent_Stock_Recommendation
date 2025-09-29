import os
import asyncio
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor
from langchain_core.messages import convert_to_messages
import uuid
from datetime import datetime
import json

load_dotenv()

app = FastAPI(title="Stock Analysis Multi-Agent API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class StockAnalysisRequest(BaseModel):
    query: str = "Analyze promising NSE stocks for short-term trading"
    session_id: str = "default"

class StockAnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    messages: List[Dict[str, Any]]
    final_recommendations: Optional[str] = None
    session_id: str

class AnalysisStatus(BaseModel):
    analysis_id: str
    status: str  # "running", "completed", "failed"
    progress: str
    session_id: str

# Global variables
supervisor = None
client = None
analysis_results = {}  # Store analysis results

def format_message_for_api(message):
    """Format message for API response"""
    try:
        return {
            "type": message.type if hasattr(message, 'type') else "unknown",
            "content": message.content if hasattr(message, 'content') else str(message),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "type": "error",
            "content": f"Error formatting message: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def extract_messages_from_update(update, last_message=False):
    """Extract and format messages from supervisor update"""
    formatted_messages = []
    
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return formatted_messages
        
        graph_id = ns[-1].split(":")[0]
        formatted_messages.append({
            "type": "subgraph_update",
            "content": f"Update from subgraph {graph_id}",
            "timestamp": datetime.now().isoformat()
        })
        is_subgraph = True

    for node_name, node_update in update.items():
        formatted_messages.append({
            "type": "node_update",
            "content": f"Update from node {node_name}",
            "agent": node_name,
            "timestamp": datetime.now().isoformat()
        })

        try:
            messages = convert_to_messages(node_update["messages"])
            if last_message:
                messages = messages[-1:]

            for m in messages:
                formatted_messages.append(format_message_for_api(m))
        except Exception as e:
            formatted_messages.append({
                "type": "error",
                "content": f"Error processing messages: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })

    return formatted_messages

async def initialize_supervisor():
    """Initialize the supervisor and agents"""
    global supervisor, client
    
    if supervisor is None:
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

        # Create agents
        stock_finder_agent = create_react_agent(
            model, tools, 
            prompt="""You are a stock research analyst specializing in the Indian Stock Market (NSE). Your task is to select 2 promising, actively traded NSE-listed stocks for short term trading (buy/sell) based on recent performance, news buzz, volume or technical strength.
            Avoid penny stocks and illiquid companies.
            Output should include stock names, tickers, and brief reasoning for each choice.
            Respond in structured plain text format.""", 
            name="stock_finder_agent"
        )

        market_data_agent = create_react_agent(
            model, tools, 
            prompt="""You are a market data analyst for Indian stocks listed on NSE. Given a list of stock tickers (eg RELIANCE, INFY), your task is to gather recent market information for each stock, including:
            - Current price
            - Previous closing price
            - Today's volume
            - 7-day and 30-day price trend
            - Basic Technical indicators (RSI, 50/200-day moving averages)
            - Any notable spikes in volume or volatility
            
            Return your findings in a structured and readable format for each stock, suitable for further analysis by a recommendation engine. Use INR as the currency. Be concise but complete.""", 
            name="market_data_agent"
        )

        news_analyst_agent = create_react_agent(
            model, tools, 
            prompt="""You are a financial news analyst. Given the names or the tickers of Indian NSE listed stocks, your job is to:
            - Search for the most recent news articles (past 3-5 days)
            - Summarize key updates, announcements, and events for each stock
            - Classify each piece of news as positive, negative or neutral
            - Highlight how the news might affect short term stock price
                                                
            Present your response in a clear, structured format - one section per stock.
            Use bullet points where necessary. Keep it short, factual and analysis-oriented""", 
            name="news_analyst_agent"
        )

        price_recommender_agent = create_react_agent(
            model, tools, 
            prompt="""You are a trading strategy advisor for the Indian Stock Market. You are given:
            - Recent market data (current price, volume, trend, indicators)
            - News summaries and sentiment for each stock
                
            Based on this info, for each stock:
            1. Recommend an action: Buy, Sell or Hold
            2. Suggest a specific target price for entry or exit (INR)
            3. Briefly explain the reason behind your recommendation.
                
            Your goal is to provide practical, near-term trading advice for the next trading day.
            Keep the response concise and clearly structured.""", 
            name="price_recommender_agent"
        )

        supervisor = create_supervisor(
            model=model,
            agents=[stock_finder_agent, market_data_agent, news_analyst_agent, price_recommender_agent],
            prompt=(
                "You are a supervisor managing four agents:\n"
                "- a stock_finder_agent. Assign research-related tasks to this agent and pick 2 promising NSE stocks\n"
                "- a market_data_agent. Assign tasks to fetch current market data (price, volume, trends)\n"
                "- a news_analyst_agent. Assign task to search and summarize recent news\n"
                "- a price_recommender_agent. Assign task to give buy/sell decision with target price.\n"
                "Assign work to one agent at a time, do not call agents in parallel.\n"
                "Do not do any work yourself.\n"
                "Make sure you complete till end and do not ask for proceed in between the task."
            ),
            add_handoff_back_messages=True,
            output_mode="full_history",
        ).compile()

async def run_stock_analysis(query: str, analysis_id: str, session_id: str):
    """Run the stock analysis and store results"""
    try:
        analysis_results[analysis_id] = {
            "status": "running",
            "messages": [],
            "session_id": session_id,
            "progress": "Starting analysis..."
        }

        if supervisor is None:
            await initialize_supervisor()

        all_messages = []
        final_chunk = None

        analysis_results[analysis_id]["progress"] = "Running multi-agent analysis..."

        for chunk in supervisor.stream({
            "messages": [{"role": "user", "content": query}]
        }):
            formatted_messages = extract_messages_from_update(chunk, last_message=True)
            all_messages.extend(formatted_messages)
            final_chunk = chunk
            
            # Update progress
            analysis_results[analysis_id]["messages"] = all_messages
            analysis_results[analysis_id]["progress"] = f"Processing... ({len(all_messages)} updates)"

        # Extract final recommendations
        final_recommendations = ""
        if final_chunk and "supervisor" in final_chunk:
            final_message_history = final_chunk["supervisor"]["messages"]
            if final_message_history:
                last_message = final_message_history[-1]
                if hasattr(last_message, 'content'):
                    final_recommendations = last_message.content

        analysis_results[analysis_id] = {
            "status": "completed",
            "messages": all_messages,
            "final_recommendations": final_recommendations,
            "session_id": session_id,
            "progress": "Analysis completed successfully"
        }

    except Exception as e:
        analysis_results[analysis_id] = {
            "status": "failed",
            "messages": [],
            "error": str(e),
            "session_id": session_id,
            "progress": f"Analysis failed: {str(e)}"
        }

@app.on_event("startup")
async def startup_event():
    """Initialize supervisor on startup"""
    await initialize_supervisor()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global client
    if client:
        # Add cleanup logic if needed
        pass

@app.get("/")
async def root():
    return {"message": "Stock Analysis Multi-Agent API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "supervisor_initialized": supervisor is not None,
        "active_analyses": len(analysis_results)
    }

@app.post("/analyze", response_model=StockAnalysisResponse)
async def start_analysis(request: StockAnalysisRequest, background_tasks: BackgroundTasks):
    """Start stock analysis (async)"""
    try:
        analysis_id = str(uuid.uuid4())
        
        # Start analysis in background
        background_tasks.add_task(
            run_stock_analysis, 
            request.query, 
            analysis_id, 
            request.session_id
        )
        
        return StockAnalysisResponse(
            analysis_id=analysis_id,
            status="started",
            messages=[],
            session_id=request.session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting analysis: {str(e)}")

@app.get("/analysis/{analysis_id}", response_model=StockAnalysisResponse)
async def get_analysis_result(analysis_id: str):
    """Get analysis result by ID"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    result = analysis_results[analysis_id]
    
    return StockAnalysisResponse(
        analysis_id=analysis_id,
        status=result["status"],
        messages=result.get("messages", []),
        final_recommendations=result.get("final_recommendations"),
        session_id=result["session_id"]
    )

@app.get("/analysis/{analysis_id}/status", response_model=AnalysisStatus)
async def get_analysis_status(analysis_id: str):
    """Get analysis status"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    result = analysis_results[analysis_id]
    
    return AnalysisStatus(
        analysis_id=analysis_id,
        status=result["status"],
        progress=result.get("progress", "Unknown"),
        session_id=result["session_id"]
    )

@app.post("/analyze/sync", response_model=StockAnalysisResponse)
async def analyze_sync(request: StockAnalysisRequest):
    """Run stock analysis synchronously (blocking)"""
    try:
        analysis_id = str(uuid.uuid4())
        
        await run_stock_analysis(request.query, analysis_id, request.session_id)
        
        result = analysis_results[analysis_id]
        
        return StockAnalysisResponse(
            analysis_id=analysis_id,
            status=result["status"],
            messages=result.get("messages", []),
            final_recommendations=result.get("final_recommendations"),
            session_id=request.session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running analysis: {str(e)}")

@app.get("/analyses")
async def list_analyses():
    """List all analyses"""
    return {
        "analyses": [
            {
                "analysis_id": aid,
                "status": result["status"],
                "session_id": result["session_id"],
                "progress": result.get("progress", "Unknown")
            }
            for aid, result in analysis_results.items()
        ]
    }

@app.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete analysis result"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    del analysis_results[analysis_id]
    return {"message": "Analysis deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)