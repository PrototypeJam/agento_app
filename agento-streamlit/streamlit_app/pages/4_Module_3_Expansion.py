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

# Check if module3.py exists
module3_path = os.path.join(module_dir, 'module3.py')
st.sidebar.write(f"module3.py exists: {os.path.exists(module3_path)}")
if os.path.exists(module3_path):
    st.sidebar.write(f"module3.py size: {os.path.getsize(module3_path)} bytes")

from utils.session_state import init_session_state, save_module_output, update_module_status, get_previous_module_output, save_logs
from utils.file_handlers import download_json, download_text, display_json

# Initialize session state
init_session_state()

st.title("📊 Module 3: Plan Expansion and Evaluation")
st.markdown("Expand plan items into detailed descriptions and evaluate them against success criteria.")

# Check API key
if not st.session_state.api_key:
    st.error("❌ Please configure your OpenAI API key in the API Configuration page first.")
    st.stop()

# Debug: Show API key status
st.sidebar.write(f"API Key set: {'Yes' if st.session_state.api_key else 'No'}")
if st.session_state.api_key:
    st.sidebar.write(f"API Key prefix: {st.session_state.api_key[:10]}...")

# Check for previous module output
previous_output = get_previous_module_output('module3')
if previous_output:
    st.success("✅ Input available from Module 2")
    
    # Display summary of Module 2 output
    st.subheader("📥 Input from Module 2")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Goal", "Set" if previous_output.get('goal') else "Missing")
    with col2:
        st.metric("Success Criteria", len(previous_output.get('selected_criteria', [])))
    with col3:
        selected_outline = previous_output.get('selected_outline', {})
        st.metric("Selected Plan", "✅ Yes" if selected_outline else "❌ No")
    with col4:
        if selected_outline:
            num_items = len(selected_outline.get('plan_items', []))
            st.metric("Plan Items", num_items)
    
    # Show the goal
    if previous_output.get('goal'):
        st.info(f"**Goal:** {previous_output['goal']}")
    
    # Show selected plan outline summary
    if selected_outline:
        st.subheader("🎯 Selected Plan to Expand")
        st.success(f"**{selected_outline.get('plan_title', 'N/A')}**")
        st.write(f"**Description:** {selected_outline.get('plan_description', 'N/A')}")
        st.write(f"**Rating:** {selected_outline.get('rating', 'N/A')}/10")
        
        # Show plan items that will be expanded
        if selected_outline.get('plan_items'):
            st.write("**Plan Items to be Expanded:**")
            for i, item in enumerate(selected_outline['plan_items'], 1):
                st.write(f"  {i}. **{item.get('item_title', 'N/A')}**: {item.get('item_description', 'N/A')}")
    
    # Show success criteria for evaluation
    if previous_output.get('selected_criteria'):
        st.subheader("🎯 Success Criteria for Evaluation")
        for i, criterion in enumerate(previous_output['selected_criteria'], 1):
            if isinstance(criterion, dict):
                st.write(f"**{i}.** {criterion.get('criteria', 'N/A')} (Rating: {criterion.get('rating', 'N/A')}/10)")
                if criterion.get('reasoning'):
                    st.caption(f"   *{criterion.get('reasoning', 'N/A')}*")
            else:
                st.write(f"**{i}.** {criterion}")
    
    with st.expander("View Full Module 2 Output JSON"):
        st.json(previous_output)
else:
    st.warning("⚠️ No output from Module 2 found. Please complete Module 2 first.")
    st.info("👈 Navigate to Module 2 using the sidebar to generate plan outlines first.")
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

# Input section for Module 3
st.markdown("---")
st.header("🔧 Configuration")

# Show what will happen
st.info("""
**Module 3 will:**
1. 📝 **Expand each plan item** into detailed, comprehensive descriptions
2. 🔍 **Evaluate each expanded item** against all success criteria  
3. 📊 **Generate evaluation results** showing pass/fail for each criterion
4. 📈 **Create summary statistics** of how well the plan meets your goals
""")

# Optional: Allow user to modify or confirm the input
confirm_input = st.checkbox("✅ Confirm input data and proceed with plan expansion and evaluation", value=True)

if not confirm_input:
    st.info("Please confirm the input data above to proceed.")
    st.stop()

# Run button
st.markdown("---")
if st.button("🚀 Run Module 3: Expand & Evaluate Plan", type="primary"):
    update_module_status('module3', 'in_progress')
    
    # Create placeholders
    status_placeholder = st.empty()
    log_container = st.container()
    
    # Debug container
    debug_container = st.expander("Debug Information", expanded=True)
    
    try:
        with st.spinner("Running Module 3..."):
            # Set the API key in environment
            os.environ['OPENAI_API_KEY'] = st.session_state.api_key
            
            with debug_container:
                st.write("Step 1: Environment setup")
                st.code(f"OPENAI_API_KEY set: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
                st.code(f"Module 2 output available: {'Yes' if previous_output else 'No'}")
                
                # Try to import the module
                st.write("Step 2: Importing module3")
                try:
                    from module3 import run_module_3
                    st.success("✅ Successfully imported run_module_3")
                    st.code(f"run_module_3 type: {type(run_module_3)}")
                    st.code(f"run_module_3 module: {run_module_3.__module__}")
                except ImportError as e:
                    st.error(f"❌ Failed to import module3: {e}")
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
            
            status_placeholder.info("🔄 Initializing Module 3...")
            
            # Define the async wrapper function
            async def run_module_async():
                with debug_container:
                    st.write("Step 4: About to call run_module_3")
                    st.code(f"Input file: {input_file}")
                    st.code(f"Output file: {output_file}")
                
                # Call the actual function with captured output
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    result = await run_module_3(input_file, output_file)
                
                with debug_container:
                    st.write("Step 5: run_module_3 completed")
                    st.code(f"Return value: {result}")
                
                return result
            
            # Show running status with progress updates
            progress_steps = [
                "🤖 Starting plan expansion...",
                "📝 Expanding plan items into detailed descriptions...",
                "🔍 Evaluating expanded items against success criteria...",
                "📊 Generating evaluation results..."
            ]
            
            for i, step in enumerate(progress_steps):
                status_placeholder.info(step)
                if i < len(progress_steps) - 1:
                    # Short delay to show progress (Module 3 takes longer)
                    import time
                    time.sleep(0.5)
            
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
                        # Show structure without full content for debug
                        structure = {k: f"<{type(v).__name__}>" for k, v in output_data.items()}
                        if 'evaluation_results' in output_data:
                            structure['evaluation_results'] = f"<list of {len(output_data['evaluation_results'])} results>"
                        st.json(structure)
                    
                    # Validate the output has expected structure
                    if not isinstance(output_data, dict):
                        raise Exception("Invalid output format: expected dictionary")
                    
                    # Save to session state
                    save_module_output('module3', output_data)
                    
                    # Get captured logs
                    stdout_log = stdout_capture.getvalue()
                    stderr_log = stderr_capture.getvalue()
                    
                    # Create standard log from captured output
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Extract key information
                    num_evaluations = len(output_data.get('evaluation_results', []))
                    expanded_outline = output_data.get('expanded_outline', {})
                    num_expanded_items = len(expanded_outline.get('plan_items', [])) if expanded_outline else 0
                    criteria_summary = output_data.get('criteria_summary', {})
                    
                    standard_log = f"""[{timestamp}] Module 3 Started
[{timestamp}] Input from Module 2: {previous_output.get('selected_outline', {}).get('plan_title', 'N/A')}
[{timestamp}] Running OpenAI Agents SDK...
[{timestamp}] Expanding {num_expanded_items} plan items...
[{timestamp}] Evaluating expanded items against success criteria...
[{timestamp}] Generated {num_evaluations} evaluation results
[{timestamp}] Criteria summary: {len(criteria_summary)} criteria evaluated
[{timestamp}] Module 3 Completed Successfully

--- Captured Output ---
{stdout_log}
""".strip()
                    
                    # Verbose log includes everything (truncated for size)
                    verbose_log = f"""{standard_log}

--- Full Output JSON (truncated) ---
{json.dumps(output_data, indent=2)[:5000]}...

--- Captured Stderr ---
{stderr_log}
""".strip()
                    
                    save_logs('module3', standard_log, verbose_log)
                    
                    # Clean up temp files
                    os.unlink(input_file)
                    os.unlink(output_file)
                    
                    status_placeholder.success("✅ Module 3 completed successfully!")
                    update_module_status('module3', 'completed')
                    
                    # Show brief summary
                    with log_container:
                        st.info(f"""
                        **Success!** Plan expansion and evaluation complete:
                        - Expanded {num_expanded_items} plan items into detailed descriptions
                        - Generated {num_evaluations} evaluation results
                        - Evaluated against {len(criteria_summary)} success criteria
                        - See detailed results below
                        """)
                else:
                    raise Exception("Output file is empty - module may have failed silently")
                
            else:
                with debug_container:
                    st.error("Output file not created!")
                    st.code(f"File exists: {os.path.exists(output_file)}")
                raise Exception("Output file not created - module may have failed")
            
    except ImportError as e:
        update_module_status('module3', 'failed')
        st.error(f"❌ Error importing module3: {str(e)}")
        st.error("Make sure module3.py is in the parent directory of streamlit_app/")
        
    except Exception as e:
        update_module_status('module3', 'failed')
        st.error(f"❌ Error running Module 3: {str(e)}")
        st.code(traceback.format_exc())
        
        # Show any captured output for debugging
        if 'stdout_capture' in locals() and stdout_capture.getvalue():
            st.text("Captured output:")
            st.code(stdout_capture.getvalue())
        if 'stderr_capture' in locals() and stderr_capture.getvalue():
            st.text("Captured errors:")
            st.code(stderr_capture.getvalue())

# Output section
if st.session_state.module_outputs.get('module3'):
    st.markdown("---")
    st.header("📤 Module 3 Output")
    
    output_data = st.session_state.module_outputs['module3']
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        expanded_outline = output_data.get('expanded_outline', {})
        num_expanded = len(expanded_outline.get('plan_items', [])) if expanded_outline else 0
        st.metric("Expanded Items", num_expanded)
    
    with col2:
        evaluation_results = output_data.get('evaluation_results', [])
        st.metric("Total Evaluations", len(evaluation_results))
    
    with col3:
        pass_count = sum(1 for result in evaluation_results if result.get('result') == 'pass')
        st.metric("Passed Evaluations", pass_count)
    
    with col4:
        fail_count = sum(1 for result in evaluation_results if result.get('result') == 'fail')
        st.metric("Failed Evaluations", fail_count)
    
    # Show criteria summary
    criteria_summary = output_data.get('criteria_summary', {})
    if criteria_summary:
        st.subheader("📊 Criteria Evaluation Summary")
        
        for criterion_name, stats in criteria_summary.items():
            col_a, col_b, col_c = st.columns([3, 1, 1])
            with col_a:
                st.write(f"**{criterion_name}**")
            with col_b:
                st.success(f"✅ Pass: {stats.get('pass', 0)}")
            with col_c:
                st.error(f"❌ Fail: {stats.get('fail', 0)}")
    
    # Show expanded plan outline
    expanded_outline = output_data.get('expanded_outline', {})
    if expanded_outline:
        st.subheader("📝 Expanded Plan Outline")
        
        st.info(f"**{expanded_outline.get('plan_title', 'N/A')}**")
        st.write(f"**Description:** {expanded_outline.get('plan_description', 'N/A')}")
        
        if expanded_outline.get('plan_items'):
            st.write("**Expanded Plan Items:**")
            for i, item in enumerate(expanded_outline['plan_items'], 1):
                with st.expander(f"📋 Step {i}: {item.get('item_title', 'N/A')}"):
                    description = item.get('item_description', 'N/A')
                    # Show first 500 characters, with option to see more
                    if len(description) > 500:
                        st.write(f"{description[:500]}...")
                        if st.button(f"Show full description for Step {i}", key=f"expand_{i}"):
                            st.write(description)
                    else:
                        st.write(description)
    
    # Show evaluation results
    if evaluation_results:
        st.subheader("🔍 Detailed Evaluation Results")
        
        # Group by criteria for better display
        criteria_groups = {}
        for result in evaluation_results:
            criteria = result.get('criteria', {})
            criteria_name = criteria.get('criteria', 'Unknown Criterion')
            if criteria_name not in criteria_groups:
                criteria_groups[criteria_name] = []
            criteria_groups[criteria_name].append(result)
        
        for criteria_name, results in criteria_groups.items():
            with st.expander(f"🎯 {criteria_name} ({len(results)} evaluations)"):
                for i, result in enumerate(results, 1):
                    result_status = result.get('result', 'unknown')
                    status_emoji = "✅" if result_status == 'pass' else "❌"
                    
                    st.write(f"**Evaluation {i}:** {status_emoji} {result_status.upper()}")
                    if result.get('reasoning'):
                        st.write(f"**Reasoning:** {result['reasoning']}")
                    st.write("---")
    
    # Display the full output JSON
    st.subheader("📄 Full Output JSON")
    display_json(output_data)
    
    # Download options
    st.subheader("📥 Downloads")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        download_json(output_data, "module3_output.json")
    
    logs = st.session_state.current_logs.get('module3', {})
    with col2:
        if logs.get('standard'):
            download_text(logs['standard'], "module3_standard.log", "📥 Download Standard Log")
    
    with col3:
        if logs.get('verbose'):
            download_text(logs['verbose'], "module3_verbose.log", "📥 Download Verbose Log")
    
    # Send to next module button
    st.markdown("---")
    if st.button("📨 Send to Module 4", type="primary"):
        st.success("✅ Output ready for Module 4!")
        st.info("Navigate to Module 4 using the sidebar to continue with revision identification.")

# Display logs if available
if st.session_state.current_logs.get('module3'):
    st.markdown("---")
    st.header("📋 Logs")
    
    logs = st.session_state.current_logs['module3']
    log_type = st.radio("Select log type:", ["Standard", "Verbose"], key="module3_log_type")
    
    if log_type == "Standard" and logs.get('standard'):
        st.text_area("Standard Log", value=logs['standard'], height=300, disabled=True, key="module3_standard_log")
    elif log_type == "Verbose" and logs.get('verbose'):
        st.text_area("Verbose Log", value=logs['verbose'], height=300, disabled=True, key="module3_verbose_log")

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

if output_data := st.session_state.module_outputs.get('module3'):
    expanded_outline = output_data.get('expanded_outline', {})
    if expanded_outline:
        st.sidebar.success(f"Expanded: {expanded_outline.get('plan_title', 'N/A')[:30]}...")
    
    # Show quick stats in sidebar
    evaluation_results = output_data.get('evaluation_results', [])
    if evaluation_results:
        pass_count = sum(1 for result in evaluation_results if result.get('result') == 'pass')
        fail_count = len(evaluation_results) - pass_count
        st.sidebar.metric("Pass/Fail", f"{pass_count}/{fail_count}")
