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
import traceback
import nest_asyncio

# THIS IS CRITICAL: Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Debug: Show current file location
st.sidebar.write("Debug Info:")
st.sidebar.code(f"Current file: {__file__}")
st.sidebar.code(f"Current dir: {os.path.dirname(__file__)}")

# Add parent directory to path
module_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(module_dir)

st.sidebar.code(f"Module dir: {module_dir}")
st.sidebar.code(f"sys.path: {sys.path[:3]}...")  # Show first 3 paths

# Check if module1.py exists
module1_path = os.path.join(module_dir, 'module1.py')
st.sidebar.write(f"module1.py exists: {os.path.exists(module1_path)}")
if os.path.exists(module1_path):
    st.sidebar.write(f"module1.py size: {os.path.getsize(module1_path)} bytes")

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

# Debug: Show API key status
st.sidebar.write(f"API Key set: {'Yes' if st.session_state.api_key else 'No'}")
if st.session_state.api_key:
    st.sidebar.write(f"API Key prefix: {st.session_state.api_key[:10]}...")

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

# Helper function to run async code in Streamlit
def run_async_function(coro):
    """Run async function safely in Streamlit environment"""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, use nest_asyncio
            return asyncio.run(coro)
        else:
            # If no loop is running, create new one
            return asyncio.run(coro)
    except RuntimeError:
        # Fallback: create new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

# Run button
if st.button("🚀 Run Module 1", type="primary", disabled=not user_goal):
    update_module_status('module1', 'in_progress')
    
    # Create placeholders
    status_placeholder = st.empty()
    log_container = st.container()
    
    # Debug container
    debug_container = st.expander("Debug Information", expanded=True)
    
    try:
        with st.spinner("Running Module 1..."):
            # Set the API key in environment
            os.environ['OPENAI_API_KEY'] = st.session_state.api_key
            
            with debug_container:
                st.write("Step 1: Environment setup")
                st.code(f"OPENAI_API_KEY set: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
                st.code(f"User goal: {user_goal}")
                
                # Try to import the module
                st.write("Step 2: Importing module1")
                try:
                    from module1 import run_module_1
                    st.success("✅ Successfully imported run_module_1")
                    st.code(f"run_module_1 type: {type(run_module_1)}")
                    st.code(f"run_module_1 module: {run_module_1.__module__}")
                except ImportError as e:
                    st.error(f"❌ Failed to import module1: {e}")
                    st.code(traceback.format_exc())
                    raise
            
            # Create a temporary output file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                output_file = tmp_file.name
                
            with debug_container:
                st.write("Step 3: Created output file")
                st.code(f"Output file: {output_file}")
            
            # Capture stdout and stderr for logs
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            status_placeholder.info("🔄 Initializing Module 1...")
            
            # Define the async wrapper function
            async def run_module_async():
                with debug_container:
                    st.write("Step 4: About to call run_module_1")
                    st.code(f"Goal being passed: {user_goal}")
                    st.code(f"Output file: {output_file}")

                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    returned_value_from_backend = await run_module_1(user_goal, output_file)

                with debug_container:
                    st.write("Step 5: run_module_1 completed")
                    st.code(f"Return value from backend run_module_1: {returned_value_from_backend}")

                return returned_value_from_backend
            
            # Show running status
            status_placeholder.info("🤖 Calling OpenAI API to generate success criteria...")
            
            # Run the coroutine using our safe async runner
            with debug_container:
                st.write("Step 6: Running async function safely")
            
            # Use our safe async runner
            trace_files_info_dict_or_none = run_async_function(run_module_async())
            
            # Check what was captured
            with debug_container:
                st.write("Step 7: Checking captured output")
                stdout_content = stdout_capture.getvalue()
                stderr_content = stderr_capture.getvalue()
                st.code(f"Stdout length: {len(stdout_content)}")
                st.code(f"Stderr length: {len(stderr_content)}")
                if stdout_content:
                    st.text_area("Captured stdout:", stdout_content, height=200)
                if stderr_content:
                    st.text_area("Captured stderr:", stderr_content, height=200)
            
            # Read the output
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    content = f.read()
                    
                with debug_container:
                    st.write("Step 8: Reading output file")
                    st.code(f"File size: {len(content)} bytes")
                    if content:
                        st.text_area("Raw file content:", content, height=200)
                    else:
                        st.error("⚠️ Output file is empty!")
                
                if content.strip():  # Only parse if file has content
                    # Parse JSON
                    output_data = json.loads(content)
                    
                    with debug_container:
                        st.write("Step 9: Parsed JSON")
                        st.json(output_data)
                    
                    # Validate the output has expected structure
                    if not isinstance(output_data, dict):
                        raise Exception("Invalid output format: expected dictionary")
                    
                    # Save to session state
                    save_module_output('module1', output_data)

                    if trace_files_info_dict_or_none:
                        if 'current_logs' not in st.session_state:
                            st.session_state.current_logs = {}
                        if 'module1' not in st.session_state.current_logs:
                            st.session_state.current_logs['module1'] = {}
                        st.session_state.current_logs['module1']['trace_files_info'] = trace_files_info_dict_or_none
                        with debug_container:
                            st.write("Debug: Stored trace_files_info for module1 into session state.")
                            st.json(trace_files_info_dict_or_none)
                    else:
                        with debug_container:
                            st.warning("Debug: No trace_files_info_dict returned or it was None. Trace files might not have been generated.")
                    
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
                    raise Exception("Output file is empty - module may have failed silently")
                
            else:
                with debug_container:
                    st.error("Output file not created!")
                    st.code(f"File exists: {os.path.exists(output_file)}")
                raise Exception("Output file not created - module may have failed")
            
    except ImportError as e:
        update_module_status('module1', 'failed')
        st.error(f"❌ Error importing module1: {str(e)}")
        st.error("Make sure module1.py is in the parent directory of streamlit_app/")
        
    except Exception as e:
        update_module_status('module1', 'failed')
        st.error(f"❌ Error running Module 1: {str(e)}")
        st.code(traceback.format_exc())
        
        # Show any captured output for debugging
        if 'stdout_capture' in locals() and stdout_capture.getvalue():
            st.text("Captured output:")
            st.code(stdout_capture.getvalue())
        if 'stderr_capture' in locals() and stderr_capture.getvalue():
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
                # Handle both string and dict formats
                if isinstance(criterion, str):
                    st.write(criterion)
                elif isinstance(criterion, dict):
                    st.write(f"**Criteria:** {criterion.get('criteria', 'N/A')}")
                    st.write(f"**Reasoning:** {criterion.get('reasoning', 'N/A')}")
                    st.write(f"**Rating:** {criterion.get('rating', 'N/A')}/10")
                else:
                    st.write(f"Unexpected data type: {type(criterion)}")
    
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

    st.markdown("---")
    st.subheader("📊 Trace File Downloads")
    trace_dl_cols = st.columns(2)
    module1_logs_session = st.session_state.current_logs.get('module1', {})
    trace_files_info_from_session = module1_logs_session.get('trace_files_info')

    if trace_files_info_from_session and isinstance(trace_files_info_from_session, dict):
        raw_sdk_path = trace_files_info_from_session.get("raw_sdk_spans_jsonl")
        if raw_sdk_path and os.path.exists(raw_sdk_path):
            with open(raw_sdk_path, 'r', encoding='utf-8') as f_raw_sdk:
                raw_sdk_content = f_raw_sdk.read()
            with trace_dl_cols[0]:
                st.download_button(
                    label="📥 Download Raw SDK Spans (JSONL)",
                    data=raw_sdk_content,
                    file_name=os.path.basename(raw_sdk_path),
                    mime='application/jsonl',
                    key=f"download_raw_sdk_{os.path.basename(raw_sdk_path)}"
                )
        elif raw_sdk_path:
            with trace_dl_cols[0]:
                st.caption(f"File not found: {os.path.basename(raw_sdk_path)}")

        eval_csv_path = trace_files_info_from_session.get("eval_data_csv")
        if eval_csv_path and os.path.exists(eval_csv_path):
            with open(eval_csv_path, 'r', encoding='utf-8') as f_eval_csv:
                eval_csv_content = f_eval_csv.read()
            with trace_dl_cols[1]:
                st.download_button(
                    label="📥 Download Eval Data (CSV)",
                    data=eval_csv_content,
                    file_name=os.path.basename(eval_csv_path),
                    mime='text/csv',
                    key=f"download_eval_csv_{os.path.basename(eval_csv_path)}"
                )
        elif eval_csv_path:
            with trace_dl_cols[1]:
                st.caption(f"File not found: {os.path.basename(eval_csv_path)}")
    else:
        st.caption("Trace files for Module 1 are not yet available or failed to generate.")
    
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
