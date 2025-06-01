import streamlit as st
from utils.session_state import init_session_state, get_previous_module_output

init_session_state()

st.title("🔧 Module 4: Revision Identification")
st.markdown("Identify needed revisions based on evaluation results.")

if not st.session_state.api_key:
    st.error("❌ Please configure your OpenAI API key in the API Configuration page first.")
    st.stop()

previous_output = get_previous_module_output('module4')
if previous_output:
    st.success("✅ Input available from Module 3")
else:
    st.warning("⚠️ No output from Module 3 found. Please complete Module 3 first.")

st.info("🚧 Module 4 functionality will be implemented in Phase 2")

# Sidebar status
st.sidebar.header("Module 4 Status")
status = st.session_state.module_status['module4']
status_emoji = {
    'not_started': '⭕',
    'in_progress': '🔄',
    'completed': '✅',
    'failed': '❌'
}.get(status, '❓')
st.sidebar.info(f"Status: {status_emoji} {status.replace('_', ' ').title()}")
