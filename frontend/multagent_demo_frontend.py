import streamlit as st
import requests
import json
import time
import asyncio
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

# Configure Streamlit page
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .agent-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .agent-working {
        background-color: #fff3cd;
        border-left-color: #ffc107;
        animation: pulse 2s infinite;
    }
    
    .agent-completed {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    
    .agent-waiting {
        background-color: #f8f9fa;
        border-left-color: #6c757d;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    
    .stock-recommendation {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

class StockAnalysisDashboard:
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.agents = {
            "stock_finder_agent": {
                "name": "Stock Finder",
                "description": "Finding promising NSE stocks",
                "icon": "🔍",
                "status": "waiting"
            },
            "market_data_agent": {
                "name": "Market Data Analyzer",
                "description": "Gathering market data and technical indicators",
                "icon": "📊",
                "status": "waiting"
            },
            "news_analyst_agent": {
                "name": "News Analyst",
                "description": "Analyzing recent news and sentiment",
                "icon": "📰",
                "status": "waiting"
            },
            "price_recommender_agent": {
                "name": "Price Recommender",
                "description": "Generating buy/sell recommendations",
                "icon": "💰",
                "status": "waiting"
            }
        }
        
    def render_header(self):
        """Render the main header"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
                <div style='text-align: center; padding: 2rem 0;'>
                    <h1 style='color: #1f77b4; margin-bottom: 0;'>📈 Stock Analysis Dashboard</h1>
                    <p style='color: #666; font-size: 1.2rem;'>AI-Powered NSE Stock Analysis & Recommendations</p>
                </div>
            """, unsafe_allow_html=True)
    
    def render_agent_status(self, agent_key: str, status: str, current_task: str = ""):
        """Render individual agent status card"""
        agent = self.agents[agent_key]
        
        status_class = f"agent-{status}"
        status_emoji = {
            "waiting": "⏳",
            "working": "🔄",
            "completed": "✅",
            "error": "❌"
        }
        
        status_text = {
            "waiting": "Waiting",
            "working": "Working",
            "completed": "Completed",
            "error": "Error"
        }
        
        st.markdown(f"""
            <div class="agent-card {status_class}">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center;">
                        <span style="font-size: 1.5rem; margin-right: 0.5rem;">{agent['icon']}</span>
                        <div>
                            <strong>{agent['name']}</strong>
                            <br>
                            <small style="color: #666;">{agent['description']}</small>
                            {f'<br><em style="color: #007bff;">{current_task}</em>' if current_task else ''}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 1.2rem;">{status_emoji[status]}</span>
                        <br>
                        <small><strong>{status_text[status]}</strong></small>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    def render_agents_panel(self):
        """Render the agents status panel"""
        st.markdown("### 🤖 Agent Status")
        
        # Create placeholders for each agent
        agent_placeholders = {}
        for agent_key in self.agents.keys():
            agent_placeholders[agent_key] = st.empty()
            
        return agent_placeholders
    
    def update_agent_status(self, placeholders: Dict, agent_key: str, status: str, task: str = ""):
        """Update agent status in real-time"""
        with placeholders[agent_key].container():
            self.render_agent_status(agent_key, status, task)
    
    def render_analysis_results(self, results: Dict):
        """Render analysis results"""
        st.markdown("### 📋 Analysis Results")
        
        if results.get('status') == 'success':
            data = results.get('data', [])
            final_message = results.get('final_message', '')
            
            # Display final recommendations
            if final_message:
                st.markdown("#### 🎯 Final Recommendations")
                st.markdown(f"""
                    <div class="stock-recommendation">
                        {final_message}
                    </div>
                """, unsafe_allow_html=True)
            
            # Display detailed agent responses
            st.markdown("#### 📝 Detailed Analysis")
            
            for i, message in enumerate(data):
                if message.get('role') == 'assistant' and message.get('content'):
                    with st.expander(f"Response {i+1}", expanded=i == len(data)-1):
                        st.write(message['content'])
        else:
            st.error("Analysis failed. Please try again.")
    
    def simulate_agent_progress(self, placeholders: Dict, query: str):
        """Simulate agent progress while making API call"""
        # Reset all agents to waiting
        for agent_key in self.agents.keys():
            self.update_agent_status(placeholders, agent_key, "waiting")
        
        # Simulate progress
        progress_steps = [
            ("stock_finder_agent", "working", "Scanning NSE stocks for opportunities..."),
            ("stock_finder_agent", "completed", "Found 2 promising stocks"),
            ("market_data_agent", "working", "Fetching current prices and technical data..."),
            ("market_data_agent", "completed", "Market data analysis complete"),
            ("news_analyst_agent", "working", "Analyzing recent news and sentiment..."),
            ("news_analyst_agent", "completed", "News analysis complete"),
            ("price_recommender_agent", "working", "Generating buy/sell recommendations..."),
            ("price_recommender_agent", "completed", "Recommendations ready")
        ]
        
        # Make API call in background
        try:
            response = requests.post(
                f"{self.api_base_url}/analyze-stocks",
                json={"query": query},
                timeout=300
            )
            
            # Simulate progress updates
            for i, (agent, status, task) in enumerate(progress_steps):
                time.sleep(2)  # Simulate processing time
                self.update_agent_status(placeholders, agent, status, task if status == "working" else "")
                
                # Update progress bar
                progress = (i + 1) / len(progress_steps)
                st.session_state.progress_bar.progress(progress)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {str(e)}")
            # Mark all working agents as error
            for agent_key in self.agents.keys():
                if self.agents[agent_key].get('status') == 'working':
                    self.update_agent_status(placeholders, agent_key, "error")
            return None
    
    def render_sidebar(self):
        """Render sidebar with controls and info"""
        st.sidebar.markdown("### ⚙️ Controls")
        
        # API Status Check
        try:
            health_response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if health_response.status_code == 200:
                st.sidebar.success("🟢 API Connected")
            else:
                st.sidebar.error("🔴 API Error")
        except:
            st.sidebar.error("🔴 API Disconnected")
        
        st.sidebar.markdown("---")
        
        # Analysis Settings
        st.sidebar.markdown("### 📊 Analysis Settings")
        
        analysis_type = st.sidebar.selectbox(
            "Analysis Type",
            ["Short-term Trading", "Long-term Investment", "Swing Trading"]
        )
        
        market_cap = st.sidebar.multiselect(
            "Market Cap Filter",
            ["Large Cap", "Mid Cap", "Small Cap"],
            default=["Large Cap", "Mid Cap"]
        )
        
        sectors = st.sidebar.multiselect(
            "Sector Preference",
            ["Technology", "Banking", "Pharma", "Auto", "FMCG", "Energy"],
            default=[]
        )
        
        st.sidebar.markdown("---")
        
        # Information Panel
        st.sidebar.markdown("### ℹ️ About")
        st.sidebar.info("""
        This dashboard uses AI agents to:
        
        🔍 **Find Stocks**: Identify promising NSE stocks
        
        📊 **Analyze Data**: Gather market data and technical indicators
        
        📰 **Check News**: Analyze recent news sentiment
        
        💰 **Recommend**: Provide buy/sell recommendations
        """)
        
        return {
            "analysis_type": analysis_type,
            "market_cap": market_cap,
            "sectors": sectors
        }
    
    def render_metrics_dashboard(self):
        """Render key metrics dashboard"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h3>2</h3>
                    <p>Stocks Analyzed</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h3>4</h3>
                    <p>AI Agents</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="metric-card">
                    <h3>NSE</h3>
                    <p>Market Focus</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
                <div class="metric-card">
                    <h3>Real-time</h3>
                    <p>Analysis</p>
                </div>
            """, unsafe_allow_html=True)

def main():
    dashboard = StockAnalysisDashboard()
    
    # Render header
    dashboard.render_header()
    
    # Render metrics
    dashboard.render_metrics_dashboard()
    
    # Create main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Render sidebar controls
        settings = dashboard.render_sidebar()
    
    with col2:
        # Main content area
        st.markdown("### 🚀 Start Analysis")
        
        # Query input
        query = st.text_area(
            "Custom Query (Optional)",
            placeholder="Enter specific requirements or leave empty for default analysis...",
            height=100
        )
        
        # Analysis button
        if st.button("🔍 Start Stock Analysis", type="primary", use_container_width=True):
            # Create progress bar
            st.session_state.progress_bar = st.progress(0)
            
            # Create agents panel
            agent_placeholders = dashboard.render_agents_panel()
            
            # Create results placeholder
            results_placeholder = st.empty()
            
            # Start analysis
            with st.spinner("Initializing AI agents..."):
                results = dashboard.simulate_agent_progress(agent_placeholders, query)
            
            # Clear progress bar
            st.session_state.progress_bar.empty()
            
            # Display results
            if results:
                with results_placeholder.container():
                    dashboard.render_analysis_results(results)
            
            # Success message
            if results and results.get('status') == 'success':
                st.balloons()
                st.success("✅ Analysis completed successfully!")
        
        # Quick action buttons
        st.markdown("### ⚡ Quick Actions")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("📈 Trending Stocks", use_container_width=True):
                st.info("Analyzing trending stocks...")
        
        with col_b:
            if st.button("💎 Value Picks", use_container_width=True):
                st.info("Finding undervalued stocks...")
        
        with col_c:
            if st.button("🔥 Momentum Plays", use_container_width=True):
                st.info("Identifying momentum stocks...")

if __name__ == "__main__":
    main()