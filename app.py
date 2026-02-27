"""
Streamlit App - Monday.com Business Intelligence Agent
A conversational AI agent for querying business data from Monday.com
"""

import streamlit as st
from agent import BIAgent
import time

# Page configuration
st.set_page_config(
    page_title="Monday.com BI Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .tool-trace {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        font-family: monospace;
        font-size: 0.85em;
    }
    .tool-trace-success {
        border-left: 4px solid #28a745;
    }
    .tool-trace-error {
        border-left: 4px solid #dc3545;
    }
    .tool-trace-executing {
        border-left: 4px solid #ffc107;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .data-quality-warning {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = BIAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "tool_traces" not in st.session_state:
    st.session_state.tool_traces = []


def render_tool_traces(traces):
    """Render tool execution traces in sidebar"""
    for trace in traces:
        status_class = f"tool-trace-{trace['status']}"
        status_emoji = {"success": "✅", "error": "❌", "executing": "⏳"}.get(trace["status"], "❓")

        st.markdown(f"""
        <div class="tool-trace {status_class}">
            {status_emoji} <strong>{trace['tool']}</strong><br>
            Status: {trace['status']}<br>
            {f"Records: {trace.get('records_fetched', 'N/A')}" if trace.get('records_fetched') else ""}
            {f"<br>Error: {trace.get('error', '')}" if trace.get('error') else ""}
        </div>
        """, unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.title("📊 BI Agent")
    st.markdown("---")

    st.markdown("### About")
    st.markdown("""
    This AI agent answers business intelligence queries by fetching **live data** from Monday.com boards:

    - **Deals Board**: Sales pipeline, deal values, stages, sectors
    - **Work Orders Board**: Projects, billing, collections, execution status
    """)

    st.markdown("---")

    st.markdown("### 🔧 Tool Execution Traces")
    if st.session_state.tool_traces:
        render_tool_traces(st.session_state.tool_traces)
    else:
        st.markdown("*No tool calls yet. Ask a question to see API traces.*")

    st.markdown("---")

    st.markdown("### Sample Questions")
    sample_questions = [
        "How's our pipeline looking?",
        "What's the total deal value by sector?",
        "Show me deals with high closure probability",
        "What's our billing status on work orders?",
        "Which sectors have the most open deals?",
        "What's the collection status for work orders?",
    ]

    for q in sample_questions:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")

    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.tool_traces = []
        st.session_state.agent.reset_conversation()
        st.rerun()


# Main chat interface
st.title("Monday.com Business Intelligence Agent")
st.markdown("Ask questions about your deals and work orders. I'll fetch live data from Monday.com to answer.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle sample question clicks
if "pending_question" in st.session_state:
    user_input = st.session_state.pending_question
    del st.session_state.pending_question

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in st.session_state.agent.chat(user_input):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    # Update tool traces
    st.session_state.tool_traces = st.session_state.agent.get_traces()

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.rerun()

# Chat input
if user_input := st.chat_input("Ask about your deals or work orders..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in st.session_state.agent.chat(user_input):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    # Update tool traces
    st.session_state.tool_traces = st.session_state.agent.get_traces()

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.rerun()


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.85em;">
    <p>Built with Streamlit • Powered by Groq LLM • Data from Monday.com</p>
    <p>All queries fetch <strong>live data</strong> - no caching</p>
</div>
""", unsafe_allow_html=True)
