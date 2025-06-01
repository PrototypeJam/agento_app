import streamlit as st
from utils.session_state import init_session_state, get_previous_module_output

# Initialize session state
init_session_state()

st.title("📋 Module 2: Plan Generation")
st.markdown("Generate and evaluate multiple plan outlines based on success criteria.")

# Check API key
if not st.session_state.api_key:
    st.error("❌ Please configure your OpenAI API key in the API Configuration page first.")
    st.stop()

# Check for previous module output
previous_output = get_previous_module_output('module2')
if previous_output:
    st.success("✅ Input available from Module 1")
    with st.expander("View Module 1 Output"):
        st.json(previous_output)
else:
    st.warning("⚠️ No output from Module 1 found. Please complete Module 1 first.")

st.info("🚧 Module 2 functionality will be implemented in Phase 2")

# Placeholder for module content
st.markdown("""
### Coming Soon:
- Load output from Module 1
- Generate multiple plan outlines
- Evaluate and rank plans
- Select the best plan
- Export results to Module 3
""")

# Sidebar status
st.sidebar.header("Module 2 Status")
status = st.session_state.module_status['module2']
status_emoji = {
    'not_started': '⭕',
    'in_progress': '🔄',
    'completed': '✅',
    'failed': '❌'
}.get(status, '❓')
st.sidebar.info(f"Status: {status_emoji} {status.replace('_', ' ').title()}")
