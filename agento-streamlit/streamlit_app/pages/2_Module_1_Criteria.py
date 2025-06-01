import streamlit as st
import json
import asyncio
import sys
import os
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.session_state import init_session_state, save_module_output, update_module_status, format_json_for_display, save_logs
from utils.file_handlers import download_json, download_text, display_json

# Initialize session state
init_session_state()

st.title("🎯 Module 1: Criteria Generation")
st.markdown("Generate success criteria for your goal or idea.")

# Check API key
if not st.session_state.api_key:
    st.error("❌ Please configure your OpenAI API key in the API Configuration page first.")
    st.stop()

# Input section
st.header("📝 Input")

input_method = st.radio("Choose input method:", ["Text Input", "JSON Input"])

if input_method == "Text Input":
    user_goal = st.text_area(
        "Enter your goal or idea:",
        placeholder="Example: Build a sustainable urban farming system...",
        height=100
    )
else:
    st.info("For MVP, please use Text Input. JSON input will be available in the next version.")
    user_goal = None

# Run button
if st.button("🚀 Run Module 1", type="primary", disabled=not user_goal):
    update_module_status('module1', 'in_progress')

    # Create placeholder for logs
    log_placeholder = st.empty()
    progress_bar = st.progress(0)

    try:
        with st.spinner("Running Module 1..."):
            # Import the module
            os.environ['OPENAI_API_KEY'] = st.session_state.api_key
            from module1 import run_module_1

            # Create a temporary output file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                output_file = tmp_file.name

            # Capture logs
            log_capture = []

            # Simple progress simulation for MVP
            for i in range(5):
                progress_bar.progress((i + 1) * 20)
                log_placeholder.text(f"Processing... Step {i+1}/5")
                import time
                time.sleep(0.5)

            # Run the module
            log_placeholder.text("Generating success criteria...")

            # Run the async function
            async def run_module():
                await run_module_1(user_goal, output_file)

            # Run the coroutine
            asyncio.run(run_module())

            # Read the output
            with open(output_file, 'r') as f:
                output_data = json.load(f)

            # Save to session state
            save_module_output('module1', output_data)

            # For MVP, create simple logs
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            standard_log = f"""
[{timestamp}] Module 1 Started
[{timestamp}] Goal: {user_goal}
[{timestamp}] Generating success criteria...
[{timestamp}] Generated {len(output_data.get('success_criteria', []))} criteria
[{timestamp}] Selected {len(output_data.get('selected_criteria', []))} top criteria
[{timestamp}] Module 1 Completed Successfully
            """.strip()

            verbose_log = standard_log + f"\n\n[{timestamp}] Full output:\n{json.dumps(output_data, indent=2)}"

            save_logs('module1', standard_log, verbose_log)

            # Clean up temp file
            os.unlink(output_file)

            progress_bar.progress(100)
            st.success("✅ Module 1 completed successfully!")
            update_module_status('module1', 'completed')

    except Exception as e:
        update_module_status('module1', 'failed')
        st.error(f"❌ Error running Module 1: {str(e)}")
        st.exception(e)

# Output section
if st.session_state.module_outputs.get('module1'):
    st.markdown("---")
    st.header("📤 Output")

    output_data = st.session_state.module_outputs['module1']

    # Display key information
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Criteria Generated", len(output_data.get('success_criteria', [])))
    with col2:
        st.metric("Selected Top Criteria", len(output_data.get('selected_criteria', [])))

    # Display the output
    st.subheader("Output JSON")
    display_json(output_data)

    # Download options
    st.subheader("📥 Downloads")
    col1, col2, col3 = st.columns(3)

    with col1:
        download_json(output_data, "module1_output.json")

    logs = st.session_state.current_logs.get('module1', {})
    with col2:
        if logs.get('standard'):
            download_text(logs['standard'], "module1_standard.log", "📥 Download Standard Log")

    with col3:
        if logs.get('verbose'):
            download_text(logs['verbose'], "module1_verbose.log", "📥 Download Verbose Log")

    # Send to next module button
    st.markdown("---")
    if st.button("📨 Send to Module 2", type="primary"):
        st.success("✅ Output ready for Module 2!")
        st.info("Navigate to Module 2 using the sidebar to continue.")

# Display logs if available
if st.session_state.current_logs.get('module1'):
    st.markdown("---")
    st.header("📋 Logs")

    logs = st.session_state.current_logs['module1']
    log_type = st.radio("Select log type:", ["Standard", "Verbose"])

    if log_type == "Standard" and logs.get('standard'):
        st.text_area("Standard Log", value=logs['standard'], height=300, disabled=True)
    elif log_type == "Verbose" and logs.get('verbose'):
        st.text_area("Verbose Log", value=logs['verbose'], height=300, disabled=True)

# Sidebar status
st.sidebar.header("Module 1 Status")
status = st.session_state.module_status['module1']
status_emoji = {
    'not_started': '⭕',
    'in_progress': '🔄',
    'completed': '✅',
    'failed': '❌'
}.get(status, '❓')
st.sidebar.info(f"Status: {status_emoji} {status.replace('_', ' ').title()}")

if output_data := st.session_state.module_outputs.get('module1'):
    st.sidebar.success(f"Goal: {output_data.get('goal', 'N/A')[:50]}...")
