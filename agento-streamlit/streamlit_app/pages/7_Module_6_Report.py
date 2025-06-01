import streamlit as st
from utils.session_state import init_session_state, get_previous_module_output

init_session_state()

st.title("📄 Module 6: Report Generation")
st.markdown("Generate a final markdown report from the revised plan.")

if not st.session_state.api_key:
    st.error("❌ Please configure your OpenAI API key in the API Configuration page first.")
    st.stop()

previous_output = get_previous_module_output('module6')
if previous_output:
    st.success("✅ Input available from Module 5")
else:
    st.warning("⚠️ No output from Module 5 found. Please complete Module 5 first.")

st.info("🚧 Module 6 functionality will be implemented in Phase 2")

st.markdown("""
### Coming Soon:
- Generate markdown report
- Preview formatted output
- Download as .md file
""")

# Sidebar status
st.sidebar.header("Module 6 Status")
status = st.session_state.module_status['module6']
status_emoji = {
    'not_started': '⭕',
    'in_progress': '🔄',
    'completed': '✅',
    'failed': '❌'
}.get(status, '❓')
st.sidebar.info(f"Status: {status_emoji} {status.replace('_', ' ').title()}")
