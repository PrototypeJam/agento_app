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

# Check if module4.py exists
module4_path = os.path.join(module_dir, 'module4.py')
st.sidebar.write(f"module4.py exists: {os.path.exists(module4_path)}")
if os.path.exists(module4_path):
    st.sidebar.write(f"module4.py size: {os.path.getsize(module4_path)} bytes")

from utils.session_state import init_session_state, save_module_output, update_module_status, get_previous_module_output, save_logs
from utils.file_handlers import download_json, download_text, display_json

# Initialize session state
init_session_state()

st.title("🔧 Module 4: Revision Identification")
st.markdown("Identify specific revisions needed to improve plan items based on evaluation results.")

# Check API key
if not st.session_state.api_key:
    st.error("❌ Please configure your OpenAI API key in the API Configuration page first.")
    st.stop()

# Debug: Show API key status
st.sidebar.write(f"API Key set: {'Yes' if st.session_state.api_key else 'No'}")
if st.session_state.api_key:
    st.sidebar.write(f"API Key prefix: {st.session_state.api_key[:10]}...")

# Check for previous module output
previous_output = get_previous_module_output('module4')
if previous_output:
    st.success("✅ Input available from Module 3")
    
    # Display summary of Module 3 output
    st.subheader("📥 Input from Module 3")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Goal", "Set" if previous_output.get('goal') else "Missing")
    with col2:
        st.metric("Success Criteria", len(previous_output.get('selected_criteria', [])))
    with col3:
        expanded_outline = previous_output.get('expanded_outline', {})
        num_items = len(expanded_outline.get('plan_items', [])) if expanded_outline else 0
        st.metric("Expanded Items", num_items)
    with col4:
        evaluation_results = previous_output.get('evaluation_results', [])
        st.metric("Evaluations", len(evaluation_results))
    
    # Show evaluation summary
    if evaluation_results:
        pass_count = sum(1 for result in evaluation_results if result.get('result') == 'pass')
        fail_count = len(evaluation_results) - pass_count
        
        st.subheader("📊 Evaluation Summary from Module 3")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.success(f"✅ **Passed:** {pass_count}")
        with col_b:
            st.error(f"❌ **Failed:** {fail_count}")
        with col_c:
            success_rate = (pass_count / len(evaluation_results)) * 100 if evaluation_results else 0
            st.info(f"📈 **Success Rate:** {success_rate:.1f}%")
    
    # Show criteria that failed (candidates for revision)
    failed_evaluations = [result for result in evaluation_results if result.get('result') == 'fail']
    if failed_evaluations:
        st.subheader("🎯 Areas Needing Improvement")
        st.warning(f"Found {len(failed_evaluations)} failed evaluations that may need revisions:")
        
        # Group failed evaluations by criteria
        failed_by_criteria = {}
        for result in failed_evaluations:
            criteria = result.get('criteria', {})
            criteria_name = criteria.get('criteria', 'Unknown Criterion')
            if criteria_name not in failed_by_criteria:
                failed_by_criteria[criteria_name] = []
            failed_by_criteria[criteria_name].append(result)
        
        for criteria_name, failures in failed_by_criteria.items():
            st.write(f"• **{criteria_name}**: {len(failures)} failure(s)")
    else:
        st.success("🎉 All evaluations passed! No revisions may be needed.")
    
    # Show the goal and plan being analyzed
    if previous_output.get('goal'):
        st.info(f"**Goal:** {previous_output['goal']}")
    
    if expanded_outline:
        st.subheader("📋 Plan Being Analyzed")
        st.write(f"**{expanded_outline.get('plan_title', 'N/A')}**")
        st.write(f"*{expanded_outline.get('plan_description', 'N/A')}*")
        
        if expanded_outline.get('plan_items'):
            with st.expander("View Plan Items"):
                for i, item in enumerate(expanded_outline['plan_items'], 1):
                    st.write(f"**{i}. {item.get('item_title', 'N/A')}**")
                    description = item.get('item_description', 'N/A')
                    # Show truncated description
                    if len(description) > 200:
                        st.write(f"   {description[:200]}...")
                    else:
                        st.write(f"   {description}")
    
    with st.expander("View Full Module 3 Output JSON"):
        st.json(previous_output)
else:
    st.warning("⚠️ No output from Module 3 found. Please complete Module 3 first.")
    st.info("👈 Navigate to Module 3 using the sidebar to expand and evaluate your plan first.")
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

# Input section for Module 4
st.markdown("---")
st.header("🔧 Configuration")

# Show what will happen
st.info("""
**Module 4 will:**
1. 🔍 **Analyze evaluation results** to identify plan items that need improvement
2. 📝 **Generate specific revision requests** for items that failed criteria
3. ⚖️ **Evaluate each revision request** to determine if it should be approved
4. 📊 **Create improvement recommendations** with impact assessments
5. 📈 **Generate coverage summary** showing before/after criteria fulfillment
""")

# Optional: Allow user to modify or confirm the input
confirm_input = st.checkbox("✅ Confirm input data and proceed with revision identification", value=True)

if not confirm_input:
    st.info("Please confirm the input data above to proceed.")
    st.stop()

# Run button
st.markdown("---")
if st.button("🚀 Run Module 4: Identify Revisions", type="primary"):
    update_module_status('module4', 'in_progress')
    
    # Create placeholders
    status_placeholder = st.empty()
    log_container = st.container()
    
    # Debug container
    debug_container = st.expander("Debug Information", expanded=True)
    
    try:
        with st.spinner("Running Module 4..."):
            # Set the API key in environment
            os.environ['OPENAI_API_KEY'] = st.session_state.api_key
            
            with debug_container:
                st.write("Step 1: Environment setup")
                st.code(f"OPENAI_API_KEY set: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
                st.code(f"Module 3 output available: {'Yes' if previous_output else 'No'}")
                
                # Try to import the module
                st.write("Step 2: Importing module4")
                try:
                    from module4 import run_module_4
                    st.success("✅ Successfully imported run_module_4")
                    st.code(f"run_module_4 type: {type(run_module_4)}")
                    st.code(f"run_module_4 module: {run_module_4.__module__}")
                except ImportError as e:
                    st.error(f"❌ Failed to import module4: {e}")
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
            
            status_placeholder.info("🔄 Initializing Module 4...")
            
            # Define the async wrapper function
            async def run_module_async():
                with debug_container:
                    st.write("Step 4: About to call run_module_4")
                    st.code(f"Input file: {input_file}")
                    st.code(f"Output file: {output_file}")
                
                # Call the actual function with captured output
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    result = await run_module_4(input_file, output_file)
                
                with debug_container:
                    st.write("Step 5: run_module_4 completed")
                    st.code(f"Return value: {result}")
                
                return result
            
            # Show running status with progress updates
            progress_steps = [
                "🔍 Analyzing evaluation results for improvement opportunities...",
                "📝 Generating specific revision requests for failed criteria...",
                "⚖️ Evaluating revision requests for approval...",
                "📊 Creating improvement recommendations and impact assessments...",
                "📈 Generating criteria coverage summary..."
            ]
            
            for i, step in enumerate(progress_steps):
                status_placeholder.info(step)
                if i < len(progress_steps) - 1:
                    # Short delay to show progress
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
                        if 'item_details' in output_data:
                            structure['item_details'] = f"<list of {len(output_data['item_details'])} items>"
                        st.json(structure)
                    
                    # Validate the output has expected structure
                    if not isinstance(output_data, dict):
                        raise Exception("Invalid output format: expected dictionary")
                    
                    # Save to session state
                    save_module_output('module4', output_data)
                    
                    # Get captured logs
                    stdout_log = stdout_capture.getvalue()
                    stderr_log = stderr_capture.getvalue()
                    
                    # Create standard log from captured output
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Extract key information
                    item_details = output_data.get('item_details', [])
                    revisions_requested = sum(1 for item in item_details if item.get('revision_request'))
                    revisions_approved = sum(1 for item in item_details 
                                           if item.get('revision_evaluation') and
                                           item.get('revision_evaluation', {}).get('approved'))
                    coverage_summary = output_data.get('criteria_coverage_summary', {})
                    
                    standard_log = f"""[{timestamp}] Module 4 Started
[{timestamp}] Input from Module 3: Analyzing {len(item_details)} plan items
[{timestamp}] Running OpenAI Agents SDK...
[{timestamp}] Identifying revision opportunities...
[{timestamp}] Generated {revisions_requested} revision requests
[{timestamp}] Approved {revisions_approved} revisions for implementation
[{timestamp}] Coverage summary: {len(coverage_summary)} criteria analyzed
[{timestamp}] Module 4 Completed Successfully

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
                    
                    save_logs('module4', standard_log, verbose_log)
                    
                    # Clean up temp files
                    os.unlink(input_file)
                    os.unlink(output_file)
                    
                    status_placeholder.success("✅ Module 4 completed successfully!")
                    update_module_status('module4', 'completed')
                    
                    # Show brief summary
                    with log_container:
                        st.info(f"""
                        **Success!** Revision identification complete:
                        - Analyzed {len(item_details)} plan items for improvement opportunities
                        - Generated {revisions_requested} revision requests
                        - Approved {revisions_approved} revisions for implementation
                        - Created criteria coverage analysis
                        - See detailed recommendations below
                        """)
                else:
                    raise Exception("Output file is empty - module may have failed silently")
                
            else:
                with debug_container:
                    st.error("Output file not created!")
                    st.code(f"File exists: {os.path.exists(output_file)}")
                raise Exception("Output file not created - module may have failed")
            
    except ImportError as e:
        update_module_status('module4', 'failed')
        st.error(f"❌ Error importing module4: {str(e)}")
        st.error("Make sure module4.py is in the parent directory of streamlit_app/")
        
    except Exception as e:
        update_module_status('module4', 'failed')
        st.error(f"❌ Error running Module 4: {str(e)}")
        st.code(traceback.format_exc())
        
        # Show any captured output for debugging
        if 'stdout_capture' in locals() and stdout_capture.getvalue():
            st.text("Captured output:")
            st.code(stdout_capture.getvalue())
        if 'stderr_capture' in locals() and stderr_capture.getvalue():
            st.text("Captured errors:")
            st.code(stderr_capture.getvalue())

# Output section
if st.session_state.module_outputs.get('module4'):
    st.markdown("---")
    st.header("📤 Module 4 Output")
    
    output_data = st.session_state.module_outputs['module4']
    
    # Display key metrics
    item_details = output_data.get('item_details', [])
    revisions_requested = sum(1 for item in item_details if item.get('revision_request'))
    revisions_approved = sum(1 for item in item_details 
                           if item.get('revision_evaluation') and 
                           item.get('revision_evaluation', {}).get('approved'))
    revisions_rejected = revisions_requested - revisions_approved
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Items Analyzed", len(item_details))
    with col2:
        st.metric("Revisions Requested", revisions_requested)
    with col3:
        st.metric("Approved", revisions_approved)
    with col4:
        st.metric("Rejected", revisions_rejected)
    
    # Show criteria coverage summary
    coverage_summary = output_data.get('criteria_coverage_summary', {})
    if coverage_summary:
        st.subheader("📊 Criteria Coverage Summary")
        
        for criterion_name, stats in coverage_summary.items():
            with st.expander(f"🎯 {criterion_name}"):
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.write("**Original Results:**")
                    st.write(f"✅ Pass: {stats.get('original_pass', 0)}")
                    st.write(f"❌ Fail: {stats.get('original_fail', 0)}")
                
                with col_b:
                    st.write("**Estimated Improvements:**")
                    improvements = stats.get('estimated_improvements', 0)
                    if improvements > 0:
                        st.success(f"📈 +{improvements} expected improvement(s)")
                    else:
                        st.info("No improvements identified")
                
                with col_c:
                    original_fail = stats.get('original_fail', 0)
                    if original_fail > 0 and improvements > 0:
                        improvement_rate = (improvements / original_fail) * 100
                        st.write("**Improvement Rate:**")
                        st.write(f"🎯 {improvement_rate:.1f}% of failures addressed")
    
    # Show item details and revision recommendations
    if item_details:
        st.subheader("📋 Revision Recommendations by Plan Item")
        
        for i, item_detail in enumerate(item_details, 1):
            item_title = item_detail.get('item_title', f'Item {i}')
            revision_request = item_detail.get('revision_request')
            revision_evaluation = item_detail.get('revision_evaluation')
            original_evaluation = item_detail.get('original_evaluation', {})
            
            # Determine status
            if revision_request and revision_evaluation:
                if revision_evaluation.get('approved'):
                    status = "✅ APPROVED FOR REVISION"
                    status_color = "success"
                else:
                    status = "❌ REVISION REJECTED"
                    status_color = "error"
            elif not revision_request:
                status = "✅ NO REVISION NEEDED"
                status_color = "success"
            else:
                status = "⚠️ PENDING EVALUATION"
                status_color = "warning"
            
            with st.expander(f"📋 {item_title} - {status}"):
                # Show original evaluation summary
                if original_evaluation:
                    st.write("**Original Evaluation Results:**")
                    for criterion, result in original_evaluation.items():
                        result_emoji = "✅" if result == "pass" else "❌"
                        st.write(f"  {result_emoji} {criterion}: {result}")
                    st.write("")
                
                if revision_request:
                    # Show revision request
                    st.write("**🔧 Revision Request:**")
                    st.info(revision_request.get('revision_request_content', 'N/A'))
                    
                    st.write("**📝 Reasoning:**")
                    st.write(revision_request.get('reasoning', 'N/A'))
                    
                    st.write("**🎯 Targeted Criteria:**")
                    targeted_criteria = revision_request.get('targeted_criteria', [])
                    for criterion in targeted_criteria:
                        st.write(f"  • {criterion}")
                    st.write("")
                    
                    if revision_evaluation:
                        # Show evaluation of the revision
                        approved = revision_evaluation.get('approved', False)
                        
                        if approved:
                            st.success("**✅ Revision Approved**")
                        else:
                            st.error("**❌ Revision Rejected**")
                        
                        st.write("**⚖️ Evaluation Reasoning:**")
                        st.write(revision_evaluation.get('reasoning', 'N/A'))
                        
                        st.write("**📊 Impact Assessment:**")
                        impact = revision_evaluation.get('impact_assessment', 'N/A')
                        st.write(impact)
                else:
                    st.success("**✅ No revision needed - all criteria were met or no viable improvements identified.**")
    
    # Display the full output JSON
    st.subheader("📄 Full Output JSON")
    display_json(output_data)
    
    # Download options
    st.subheader("📥 Downloads")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        download_json(output_data, "module4_output.json")
    
    logs = st.session_state.current_logs.get('module4', {})
    with col2:
        if logs.get('standard'):
            download_text(logs['standard'], "module4_standard.log", "📥 Download Standard Log")
    
    with col3:
        if logs.get('verbose'):
            download_text(logs['verbose'], "module4_verbose.log", "📥 Download Verbose Log")
    
    # Send to next module button
    st.markdown("---")
    if revisions_approved > 0:
        st.info(f"🔧 {revisions_approved} revision(s) approved for implementation in Module 5.")
    else:
        st.info("ℹ️ No revisions approved - plan may proceed as-is to Module 6.")
    
    if st.button("📨 Send to Module 5", type="primary"):
        st.success("✅ Output ready for Module 5!")
        st.info("Navigate to Module 5 using the sidebar to implement approved revisions.")

# Display logs if available
if st.session_state.current_logs.get('module4'):
    st.markdown("---")
    st.header("📋 Logs")
    
    logs = st.session_state.current_logs['module4']
    log_type = st.radio("Select log type:", ["Standard", "Verbose"], key="module4_log_type")
    
    if log_type == "Standard" and logs.get('standard'):
        st.text_area("Standard Log", value=logs['standard'], height=300, disabled=True, key="module4_standard_log")
    elif log_type == "Verbose" and logs.get('verbose'):
        st.text_area("Verbose Log", value=logs['verbose'], height=300, disabled=True, key="module4_verbose_log")

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

if output_data := st.session_state.module_outputs.get('module4'):
    item_details = output_data.get('item_details', [])
    revisions_approved = sum(1 for item in item_details 
                           if item.get('revision_evaluation', {}).get('approved'))
    
    if revisions_approved > 0:
        st.sidebar.success(f"✅ {revisions_approved} revision(s) approved")
    else:
        st.sidebar.info("ℹ️ No revisions needed")
    
    # Show coverage improvement estimate
    coverage_summary = output_data.get('criteria_coverage_summary', {})
    total_improvements = sum(stats.get('estimated_improvements', 0) 
                           for stats in coverage_summary.values())
    if total_improvements > 0:
        st.sidebar.metric("Est. Improvements", f"+{total_improvements}")
