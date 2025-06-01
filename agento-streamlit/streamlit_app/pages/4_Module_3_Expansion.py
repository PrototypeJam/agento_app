import streamlit as st
from utils.session_state import init_session_state, get_previous_module_output

init_session_state()

st.title("📊 Module 3: Plan Expansion and Evaluation")
st.markdown("Expand plan items and evaluate against success criteria.")

if not st.session_state.api_key:
    st.error("❌ Please configure your OpenAI API key in the API Configuration page first.")
    st.stop()

previous_output = get_previous_module_output('module3')
if previous_output:
    st.success("✅ Input available from Module 2")
else:
    st.warning("⚠️ No output from Module 2 found. Please complete Module 2 first.")

st.info("🚧 Module 3 functionality will be implemented in Phase 2")

# Sidebar status
st.sidebar.header("Module 3 Status")
status = st.session_state.module_status['module3']
status_emoji = {
    'not_started': '⭕',
    'in_progress': '🔄',
    'completed': '✅',
    'failed': '❌'
}.get(status, '❓')
st.sidebar.info(f"Status: {status_emoji} {status.replace('_', ' ').title()}")
