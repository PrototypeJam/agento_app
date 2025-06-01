import streamlit as st
import json
import asyncio
import sys
import os
import logging
from datetime import datetime
import tempfile
import io
from contextlib import redirect_stdout, redirect_stderr

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
    
    # Create placeholders
    status_placeholder = st.empty()
    log_container = st.container()
    
    try:
        with st.spinner("Running Module 1..."):
            # Set the API key in environment
            os.environ['OPENAI_API_KEY'] = st.session_state.api_key
            
            # Import the module here to ensure API key is set
            from module1 import run_module_1
            
            # Create a temporary output file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                output_file = tmp_file.name
            
            # Capture stdout and stderr for logs
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            status_placeholder.info("🔄 Initializing Module 1...")
            
            # Run the actual module with output capture
            async def run_module():
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    await run_module_1(user_goal, output_file)
            
            # Show running status
            status_placeholder.info("🤖 Calling OpenAI API to generate success criteria...")
            
            # Run the coroutine
            asyncio.run(run_module())
            
            # Read the output
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    output_data = json.load(f)
                
                # Save to session state
                save_module_output('module1', output_data)
                
                # Get captured logs
                stdout_log = stdout_capture.getvalue()
                stderr_log = stderr_capture.getvalue()
                
                # Create standard log from captured output
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Extract key information
                num_criteria = len(output_data.get('success_criteria', []))
                num_selected = len(output_data.get('selected_criteria', []))
                
                standard_log = f"""[{timestamp}] Module 1 Started
[{timestamp}] Goal: {user_goal}
[{timestamp}] Running OpenAI Agents SDK...
[{timestamp}] Generating success criteria...
[{timestamp}] Generated {num_criteria} criteria
[{timestamp}] Selected {num_selected} top criteria
[{timestamp}] Module 1 Completed Successfully

--- Captured Output ---
{stdout_log}
""".strip()
                
                # Verbose log includes everything
                verbose_log = f"""{standard_log}

--- Full Output JSON ---
{json.dumps(output_data, indent=2)}

--- Captured Stderr ---
{stderr_log}
""".strip()
                
                save_logs('module1', standard_log, verbose_log)
                
                # Clean up temp file
                os.unlink(output_file)
                
                status_placeholder.success("✅ Module 1 completed successfully!")
                update_module_status('module1', 'completed')
                
                # Show brief summary
                with log_container:
                    st.info(f"""
                    **Success!** Generated {num_criteria} success criteria:
                    - Selected top {num_selected} criteria for evaluation
                    - See output below for details
                    """)
                
            else:
                raise Exception("Output file not created - module may have failed")
            
    except ImportError as e:
        update_module_status('module1', 'failed')
        st.error(f"❌ Error importing module1: {str(e)}")
        st.error("Make sure module1.py is in the parent directory of streamlit_app/")
        
    except Exception as e:
        update_module_status('module1', 'failed')
        st.error(f"❌ Error running Module 1: {str(e)}")
        
        # Show any captured output for debugging
        if stdout_capture.getvalue():
            st.text("Captured output:")
            st.code(stdout_capture.getvalue())
        if stderr_capture.getvalue():
            st.text("Captured errors:")
            st.code(stderr_capture.getvalue())

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
    
    # Show the actual criteria
    if output_data.get('selected_criteria'):
        st.subheader("Selected Success Criteria")
        for i, criterion in enumerate(output_data['selected_criteria'], 1):
            with st.expander(f"Criterion {i}"):
                st.write(f"**Criteria:** {criterion.get('criteria', 'N/A')}")
                st.write(f"**Reasoning:** {criterion.get('reasoning', 'N/A')}")
                st.write(f"**Rating:** {criterion.get('rating', 'N/A')}/10")
    
    # Display the full output
    st.subheader("Full Output JSON")
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