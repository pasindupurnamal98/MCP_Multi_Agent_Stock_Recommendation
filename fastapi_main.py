import os
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

load_dotenv()

app = FastAPI(title="Web Search Agent API", version="1.0.0")

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

# Global variables to store agent and client
agent = None
client = None

async def initialize_agent():
    """Initialize the MCP client and agent"""
    global agent, client
    
    if agent is None:
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
            model="azure_openai:gpt-4",
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

@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    await initialize_agent()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global client
    if client:
        # Add any cleanup logic if needed
        pass

@app.get("/")
async def root():
    return {"message": "Web Search Agent API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent_initialized": agent is not None}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for frontend integration"""
    try:
        if agent is None:
            await initialize_agent()
        
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Process the user query
        agent_response = await agent.ainvoke({
            "messages": [{"role": "user", "content": request.message}]
        })

        response_content = agent_response["messages"][-1].content

        return ChatResponse(
            response=response_content,
            session_id=request.session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Optional: Add streaming endpoint for real-time responses
@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streaming chat endpoint (if you want to implement streaming)"""
    try:
        if agent is None:
            await initialize_agent()
        
        # For now, return the same response as non-streaming
        # You can implement actual streaming if your agent supports it
        agent_response = await agent.ainvoke({
            "messages": [{"role": "user", "content": request.message}]
        })

        response_content = agent_response["messages"][-1].content
        
        return {"response": response_content, "session_id": request.session_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Optional: Get available tools
@app.get("/tools")
async def get_tools():
    """Get available tools from the MCP client"""
    try:
        if client is None:
            await initialize_agent()
        
        tools = await client.get_tools()
        return {"tools": [tool.name for tool in tools]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting tools: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)