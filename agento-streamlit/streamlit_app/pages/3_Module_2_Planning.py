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

# Apply nest_asyncio to allow nested event loops
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

# Check if module2.py exists
module2_path = os.path.join(module_dir, 'module2.py')
st.sidebar.write(f"module2.py exists: {os.path.exists(module2_path)}")
if os.path.exists(module2_path):
    st.sidebar.write(f"module2.py size: {os.path.getsize(module2_path)} bytes")

from utils.session_state import init_session_state, save_module_output, update_module_status, get_previous_module_output, save_logs
from utils.file_handlers import download_json, download_text, display_json

# Initialize session state
init_session_state()

st.title("📋 Module 2: Plan Generation")
st.markdown("Generate and evaluate multiple plan outlines based on success criteria.")

# Check API key
if not st.session_state.api_key:
    st.error("❌ Please configure your OpenAI API key in the API Configuration page first.")
    st.stop()

# Debug: Show API key status
st.sidebar.write(f"API Key set: {'Yes' if st.session_state.api_key else 'No'}")
if st.session_state.api_key:
    st.sidebar.write(f"API Key prefix: {st.session_state.api_key[:10]}...")

# Check for previous module output
previous_output = get_previous_module_output('module2')
if previous_output:
    st.success("✅ Input available from Module 1")
    
    # Display summary of Module 1 output
    st.subheader("📥 Input from Module 1")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Goal", "Set" if previous_output.get('goal') else "Missing")
    with col2:
        st.metric("Total Criteria", len(previous_output.get('success_criteria', [])))
    with col3:
        st.metric("Selected Criteria", len(previous_output.get('selected_criteria', [])))
    
    # Show the goal and selected criteria
    if previous_output.get('goal'):
        st.info(f"**Goal:** {previous_output['goal']}")
    
    if previous_output.get('selected_criteria'):
        with st.expander("View Selected Success Criteria"):
            for i, criterion in enumerate(previous_output['selected_criteria'], 1):
                if isinstance(criterion, dict):
                    st.write(f"**{i}.** {criterion.get('criteria', 'N/A')}")
                    st.write(f"   *Reasoning:* {criterion.get('reasoning', 'N/A')}")
                    st.write(f"   *Rating:* {criterion.get('rating', 'N/A')}/10")
                else:
                    st.write(f"**{i}.** {criterion}")
                st.write("")
    
    with st.expander("View Full Module 1 Output JSON"):
        st.json(previous_output)
else:
    st.warning("⚠️ No output from Module 1 found. Please complete Module 1 first.")
    st.info("👈 Navigate to Module 1 using the sidebar to generate success criteria first.")
    st.stop()

# Helper function to run async code safely
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

# Input section for Module 2
st.markdown("---")
st.header("🔧 Configuration")

# Optional: Allow user to modify or confirm the input
confirm_input = st.checkbox("✅ Confirm input data and proceed with plan generation", value=True)

if not confirm_input:
    st.info("Please confirm the input data above to proceed.")
    st.stop()

# Run button
st.markdown("---")
if st.button("🚀 Run Module 2: Generate Plans", type="primary"):
    update_module_status('module2', 'in_progress')
    
    # Create placeholders
    status_placeholder = st.empty()
    log_container = st.container()
    
    # Debug container
    debug_container = st.expander("Debug Information", expanded=True)
    
    try:
        with st.spinner("Running Module 2..."):
            # Set the API key in environment
            os.environ['OPENAI_API_KEY'] = st.session_state.api_key
            
            with debug_container:
                st.write("Step 1: Environment setup")
                st.code(f"OPENAI_API_KEY set: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
                st.code(f"Module 1 output available: {'Yes' if previous_output else 'No'}")
                
                # Try to import the module
                st.write("Step 2: Importing module2")
                try:
                    from module2 import run_module_2
                    st.success("✅ Successfully imported run_module_2")
                    st.code(f"run_module_2 type: {type(run_module_2)}")
                    st.code(f"run_module_2 module: {run_module_2.__module__}")
                except ImportError as e:
                    st.error(f"❌ Failed to import module2: {e}")
                    st.code(traceback.format_exc())
                    raise
            
            # Create temporary input and output files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_input_file:
                input_file = tmp_input_file.name
                json.dump(previous_output, tmp_input_file, indent=2)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_output_file:
                output_file = tmp_output_file.name
                
            with debug_container:
                st.write("Step 3: Created temporary files")
                st.code(f"Input file: {input_file}")
                st.code(f"Output file: {output_file}")
            
            # Capture stdout and stderr for logs
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            status_placeholder.info("🔄 Initializing Module 2...")
            
            # Define the async wrapper function
            async def run_module_async():
                with debug_container:
                    st.write("Step 4: About to call run_module_2")
                    st.code(f"Input file: {input_file}")
                    st.code(f"Output file: {output_file}")
                
                # Call the actual function with captured output
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    result = await run_module_2(input_file, output_file)
                
                with debug_container:
                    st.write("Step 5: run_module_2 completed")
                    st.code(f"Return value: {result}")
                
                return result
            
            # Show running status
            status_placeholder.info("🤖 Generating multiple plan outlines...")
            
            # Run the coroutine using our safe async runner
            with debug_container:
                st.write("Step 6: Running async function safely")
            
            # Use our safe async runner
            result = run_async_function(run_module_async())
            
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
                        st.text_area("Raw file content (first 1000 chars):", content[:1000], height=200)
                    else:
                        st.error("⚠️ Output file is empty!")
                
                if content.strip():  # Only parse if file has content
                    # Parse JSON
                    output_data = json.loads(content)
                    
                    with debug_container:
                        st.write("Step 9: Parsed JSON")
                        st.json({k: f"<{type(v).__name__}>" for k, v in output_data.items()})  # Show structure without full content
                    
                    # Validate the output has expected structure
                    if not isinstance(output_data, dict):
                        raise Exception("Invalid output format: expected dictionary")
                    
                    # Save to session state
                    save_module_output('module2', output_data)
                    
                    # Get captured logs
                    stdout_log = stdout_capture.getvalue()
                    stderr_log = stderr_capture.getvalue()
                    
                    # Create standard log from captured output
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Extract key information
                    num_outlines = len(output_data.get('plan_outlines', []))
                    selected_outline = output_data.get('selected_outline', {})
                    selected_title = selected_outline.get('plan_title', 'N/A') if selected_outline else 'N/A'
                    
                    standard_log = f"""[{timestamp}] Module 2 Started
[{timestamp}] Input from Module 1: {previous_output.get('goal', 'N/A')[:100]}...
[{timestamp}] Running OpenAI Agents SDK...
[{timestamp}] Generating plan outlines...
[{timestamp}] Generated {num_outlines} plan outlines
[{timestamp}] Selected best outline: {selected_title}
[{timestamp}] Module 2 Completed Successfully

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
                    
                    save_logs('module2', standard_log, verbose_log)
                    
                    # Clean up temp files
                    os.unlink(input_file)
                    os.unlink(output_file)
                    
                    status_placeholder.success("✅ Module 2 completed successfully!")
                    update_module_status('module2', 'completed')
                    
                    # Show brief summary
                    with log_container:
                        st.info(f"""\
                        **Success!** Generated {num_outlines} plan outlines:
                        - Selected best outline: "{selected_title}"
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
        update_module_status('module2', 'failed')
        st.error(f"❌ Error importing module2: {str(e)}")
        st.error("Make sure module2.py is in the parent directory of streamlit_app/")
        
    except Exception as e:
        update_module_status('module2', 'failed')
        st.error(f"❌ Error running Module 2: {str(e)}")
        st.code(traceback.format_exc())
        
        # Show any captured output for debugging
        if 'stdout_capture' in locals() and stdout_capture.getvalue():
            st.text("Captured output:")
            st.code(stdout_capture.getvalue())
        if 'stderr_capture' in locals() and stderr_capture.getvalue():
            st.text("Captured errors:")
            st.code(stderr_capture.getvalue())

# Output section
if st.session_state.module_outputs.get('module2'):
    st.markdown("---")
    st.header("📤 Module 2 Output")
    
    output_data = st.session_state.module_outputs['module2']
    
    # Display key information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Plan Outlines Generated", len(output_data.get('plan_outlines', [])))
    with col2:
        selected_outline = output_data.get('selected_outline', {})
        st.metric("Selected Plan", "✅ Yes" if selected_outline else "❌ No")
    with col3:
        if selected_outline:
            num_items = len(selected_outline.get('plan_items', []))
            st.metric("Plan Items", num_items)
    
    # Show the selected plan outline
    if selected_outline:
        st.subheader("🎯 Selected Plan Outline")
        
        st.info(f"**Plan Title:** {selected_outline.get('plan_title', 'N/A')}")
        st.write(f"**Description:** {selected_outline.get('plan_description', 'N/A')}")
        st.write(f"**Rating:** {selected_outline.get('rating', 'N/A')}/10")
        st.write(f"**Created by:** {selected_outline.get('created_by', 'N/A')}")
        
        if selected_outline.get('reasoning'):
            with st.expander("View Reasoning"):
                st.write(selected_outline['reasoning'])
        
        # Show plan items
        if selected_outline.get('plan_items'):
            st.subheader("📋 Plan Items")
            for i, item in enumerate(selected_outline['plan_items'], 1):
                with st.expander(f"Step {i}: {item.get('item_title', 'N/A')}"):
                    st.write(f"**Description:** {item.get('item_description', 'N/A')}")
    
    # Show all generated outlines
    if output_data.get('plan_outlines'):
        st.subheader("📊 All Generated Plan Outlines")
        for i, outline in enumerate(output_data['plan_outlines'], 1):
            is_selected = outline == selected_outline
            status = "🎯 SELECTED" if is_selected else f"#{i}"
            
            with st.expander(f"{status}: {outline.get('plan_title', f'Plan {i}')} (Rating: {outline.get('rating', 'N/A')}/10)"):
                st.write(f"**Description:** {outline.get('plan_description', 'N/A')}")
                st.write(f"**Created by:** {outline.get('created_by', 'N/A')}")
                st.write(f"**Rating:** {outline.get('rating', 'N/A')}/10")
                
                if outline.get('reasoning'):
                    st.write(f"**Reasoning:** {outline['reasoning']}")
                
                if outline.get('plan_items'):
                    st.write("**Plan Items:**")
                    for j, item in enumerate(outline['plan_items'], 1):
                        st.write(f"  {j}. **{item.get('item_title', 'N/A')}**: {item.get('item_description', 'N/A')}")
    
    # Display the full output JSON
    st.subheader("📄 Full Output JSON")
    display_json(output_data)
    
    # Download options
    st.subheader("📥 Downloads")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        download_json(output_data, "module2_output.json")
    
    logs = st.session_state.current_logs.get('module2', {})
    with col2:
        if logs.get('standard'):
            download_text(logs['standard'], "module2_standard.log", "📥 Download Standard Log")
    
    with col3:
        if logs.get('verbose'):
            download_text(logs['verbose'], "module2_verbose.log", "📥 Download Verbose Log")
    
    # Send to next module button
    st.markdown("---")
    if st.button("📨 Send to Module 3", type="primary"):
        st.success("✅ Output ready for Module 3!")
        st.info("Navigate to Module 3 using the sidebar to continue with plan expansion.")

# Display logs if available
if st.session_state.current_logs.get('module2'):
    st.markdown("---")
    st.header("📋 Logs")
    
    logs = st.session_state.current_logs['module2']
    log_type = st.radio("Select log type:", ["Standard", "Verbose"], key="module2_log_type")
    
    if log_type == "Standard" and logs.get('standard'):
        st.text_area("Standard Log", value=logs['standard'], height=300, disabled=True, key="module2_standard_log")
    elif log_type == "Verbose" and logs.get('verbose'):
        st.text_area("Verbose Log", value=logs['verbose'], height=300, disabled=True, key="module2_verbose_log")

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

if output_data := st.session_state.module_outputs.get('module2'):
    selected_plan = output_data.get('selected_outline', {})
    if selected_plan:
        st.sidebar.success(f"Selected: {selected_plan.get('plan_title', 'N/A')[:30]}...")
