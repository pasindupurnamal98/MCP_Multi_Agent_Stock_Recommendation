📈 AI-Powered Stock Analysis Dashboard
An end-to-end multi-agent stock analysis system for the Indian Stock Market (NSE).
It uses a FastAPI backend (AI agents via LangChain + LangGraph) and a Streamlit frontend (interactive dashboard) to generate trading insights.

🚀 Features
FastAPI Backend

4 specialized agents: 🔍 Stock Finder, 📊 Market Data, 📰 News Analyst, 💰 Price Recommender
Orchestrated by a LangGraph supervisor
Powered by Azure OpenAI GPT‑4 + Bright Data MCP tools
Streamlit Frontend

Real-time agent status simulation
Final Buy/Sell/Hold recommendations
Filters for strategy, market cap, and sectors
🛠 Tech Stack
FastAPI, LangChain, LangGraph
Azure OpenAI, Bright Data MCP
Streamlit, Plotly, Pandas
⚡ Quick Start
Bash

# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
API runs at: http://localhost:8000
Dashboard at: http://localhost:8501
🧠 How It Works
User query → Supervisor agent
Tasks delegated: Stock Finder → Market Data → News → Price Recommender
Agents return insights → Final structured recommendation
Streamlit displays real-time progress + results
🙏 Credits
Inspired by this [YouTube video on](https://www.youtube.com/watch?v=NF2aRqIlYNE) MCP | Multi-Agent Stock Recommendation Project.

📜 License
MIT License — free to use, fork, and modify.
