# cd agento-streamlit 
# cd streamlit_app

# uv venv
# source .venv/bin/activate
# uv pip install -r requirements.txt
# streamlit run app.py


import streamlit as st
import os
import sys

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Agento - Multi-Agent Planning System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'module_outputs' not in st.session_state:
    st.session_state.module_outputs = {}
if 'module_status' not in st.session_state:
    st.session_state.module_status = {
        'module1': 'not_started',
        'module2': 'not_started',
        'module3': 'not_started',
        'module4': 'not_started',
        'module5': 'not_started',
        'module6': 'not_started'
    }

# Main page content
st.title("🤖 Agento - Multi-Agent Planning System")
st.markdown("---")

# Dashboard view
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Modules", "6")

with col2:
    completed = sum(1 for status in st.session_state.module_status.values() if status == 'completed')
    st.metric("Completed", f"{completed}/6")

with col3:
    api_status = "✅ Set" if st.session_state.api_key else "❌ Not Set"
    st.metric("API Key Status", api_status)

st.markdown("---")

# Module Status Overview
st.header("📊 Module Status")

status_cols = st.columns(3)
modules = [
    ("Module 1", "Criteria Generation", st.session_state.module_status['module1']),
    ("Module 2", "Plan Generation", st.session_state.module_status['module2']),
    ("Module 3", "Plan Expansion", st.session_state.module_status['module3']),
    ("Module 4", "Revision Identification", st.session_state.module_status['module4']),
    ("Module 5", "Revision Implementation", st.session_state.module_status['module5']),
    ("Module 6", "Report Generation", st.session_state.module_status['module6'])
]

for idx, (name, desc, status) in enumerate(modules):
    col = status_cols[idx % 3]
    with col:
        status_emoji = {
            'not_started': '⭕',
            'in_progress': '🔄',
            'completed': '✅',
            'failed': '❌'
        }.get(status, '❓')

        st.info(f"{status_emoji} **{name}**  \n{desc}")

st.markdown("---")

# Instructions
st.header("🚀 Getting Started")
st.markdown("""
1. **Configure API Key**: Go to the API Configuration page to set your OpenAI API key
2. **Start with Module 1**: Input your goal or idea to generate success criteria
3. **Progress through Modules**: Each module's output becomes the input for the next
4. **Generate Final Report**: Module 6 creates a markdown report from your refined plan

### Navigation
Use the sidebar to navigate between modules. Each module allows you to:
- View and download outputs
- Download logs
- Send outputs to the next module
""")

# Sidebar info
st.sidebar.header("About Agento")
st.sidebar.info("""
Agento is a modular agent-based planning system that transforms your initial goal into a well-structured plan through iterative refinement and evaluation.

Each module uses specialized AI agents to handle different aspects of the planning process.
""")

# Check API key warning
if not st.session_state.api_key:
    st.warning("⚠️ Please configure your OpenAI API key in the API Configuration page to begin.")
