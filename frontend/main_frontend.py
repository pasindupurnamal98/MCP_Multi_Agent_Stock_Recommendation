import streamlit as st
import requests
import json
import time
from datetime import datetime
import uuid

# Configure the page
st.set_page_config(
    page_title="Web Search Agent Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "api_status" not in st.session_state:
    st.session_state.api_status = None

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API returned status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

def send_message_to_api(message, session_id):
    """Send message to the FastAPI backend"""
    try:
        payload = {
            "message": message,
            "session_id": session_id
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return False, f"API Error: {error_detail}"
            
    except requests.exceptions.Timeout:
        return False, "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"
    except json.JSONDecodeError:
        return False, "Invalid response from API"

def get_available_tools():
    """Get available tools from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/tools", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Error getting tools: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

# Sidebar
with st.sidebar:
    st.title("🤖 Agent Settings")
    
    # API Status
    st.subheader("API Status")
    if st.button("Check API Health", type="secondary"):
        with st.spinner("Checking API..."):
            is_healthy, status_info = check_api_health()
            st.session_state.api_status = (is_healthy, status_info)
    
    if st.session_state.api_status:
        is_healthy, status_info = st.session_state.api_status
        if is_healthy:
            st.success("✅ API is running")
            if isinstance(status_info, dict):
                st.json(status_info)
        else:
            st.error(f"❌ API Error: {status_info}")
    
    # Session Info
    st.subheader("Session Info")
    st.text(f"Session ID: {st.session_state.session_id[:8]}...")
    
    if st.button("New Session", type="secondary"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    # Available Tools
    st.subheader("Available Tools")
    if st.button("Get Tools", type="secondary"):
        with st.spinner("Getting tools..."):
            success, tools_info = get_available_tools()
            if success:
                st.success("Tools loaded!")
                if "tools" in tools_info:
                    for tool in tools_info["tools"]:
                        st.text(f"• {tool}")
                else:
                    st.json(tools_info)
            else:
                st.error(f"Error: {tools_info}")
    
    # Clear Chat
    st.subheader("Chat Controls")
    if st.button("Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("🌐 Web Search Agent Chat")
st.markdown("Ask me to search for flights, weather, or any web information!")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(f"*{message['timestamp']}*")

# Chat input
if prompt := st.chat_input("Ask me anything about flights, weather, or web search..."):
    # Add user message to chat history
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "timestamp": timestamp
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"*{timestamp}*")
    
    # Get bot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Show thinking indicator
        with st.spinner("🤔 Thinking..."):
            success, response_data = send_message_to_api(prompt, st.session_state.session_id)
        
        if success:
            bot_response = response_data.get("response", "No response received")
            message_placeholder.markdown(bot_response)
            
            # Add assistant response to chat history
            response_timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": bot_response,
                "timestamp": response_timestamp
            })
            st.caption(f"*{response_timestamp}*")
            
        else:
            error_message = f"❌ Error: {response_data}"
            message_placeholder.error(error_message)
            
            # Add error to chat history
            error_timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_message,
                "timestamp": error_timestamp
            })

# Footer with example prompts
st.markdown("---")
st.subheader("💡 Example Prompts")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🛫 Find flights from NYC to London", use_container_width=True):
        st.session_state.example_prompt = "Find flights from New York to London"

with col2:
    if st.button("🌤️ What's the weather in Paris?", use_container_width=True):
        st.session_state.example_prompt = "What's the weather like in Paris today?"

with col3:
    if st.button("🔍 Search for hotels in Tokyo", use_container_width=True):
        st.session_state.example_prompt = "Find hotels in Tokyo for next week"

# Handle example prompt clicks
if "example_prompt" in st.session_state:
    # Trigger the chat input with the example prompt
    st.session_state.messages.append({
        "role": "user", 
        "content": st.session_state.example_prompt,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    
    # Remove the example prompt from session state
    del st.session_state.example_prompt
    st.rerun()

# Additional features section
with st.expander("ℹ️ About this Agent"):
    st.markdown("""
    This Web Search Agent uses Bright Data tools to fetch real-time information including:
    
    - **Flight Information**: Search for flights, prices, and schedules
    - **Weather Data**: Get current weather conditions and forecasts
    - **Web Scraping**: Extract information from various websites
    - **Real-time Data**: Access up-to-date information from the web
    
    **How to use:**
    1. Make sure the FastAPI backend is running on `http://localhost:8000`
    2. Type your question in the chat input
    3. The agent will use appropriate tools to fetch the information
    4. View the results in the chat interface
    
    **Tips:**
    - Be specific in your queries for better results
    - Check the API status in the sidebar if you encounter issues
    - Use the example prompts to get started
    """)

# Debug section (only show if there are errors)
if any("Error:" in msg["content"] for msg in st.session_state.messages):
    with st.expander("🔧 Debug Information"):
        st.markdown("**Recent Errors:**")
        for msg in reversed(st.session_state.messages):
            if "Error:" in msg["content"]:
                st.code(msg["content"])
                break
        
        st.markdown("**Troubleshooting:**")
        st.markdown("""
        1. Ensure FastAPI server is running: `python your_fastapi_file.py`
        2. Check if the API is accessible at `http://localhost:8000`
        3. Verify your environment variables are set correctly
        4. Check the FastAPI logs for detailed error information
        """)