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

# Check if module5.py exists
module5_path = os.path.join(module_dir, 'module5.py')
st.sidebar.write(f"module5.py exists: {os.path.exists(module5_path)}")
if os.path.exists(module5_path):
    st.sidebar.write(f"module5.py size: {os.path.getsize(module5_path)} bytes")

from utils.session_state import init_session_state, save_module_output, update_module_status, get_previous_module_output, save_logs
from utils.file_handlers import download_json, download_text, display_json

# Initialize session state
init_session_state()

st.title("✨ Module 5: Revision Implementation")
st.markdown("Implement approved revisions to improve the plan and create the final refined version.")

# Check API key
if not st.session_state.api_key:
    st.error("❌ Please configure your OpenAI API key in the API Configuration page first.")
    st.stop()

# Debug: Show API key status
st.sidebar.write(f"API Key set: {'Yes' if st.session_state.api_key else 'No'}")
if st.session_state.api_key:
    st.sidebar.write(f"API Key prefix: {st.session_state.api_key[:10]}...")

# Check for previous module output
previous_output = get_previous_module_output('module5')
if previous_output:
    st.success("✅ Input available from Module 4")
    
    # Display summary of Module 4 output
    st.subheader("📥 Input from Module 4")
    
    # Get key information
    item_details = previous_output.get('item_details', [])
    revisions_requested = sum(1 for item in item_details if item.get('revision_request'))
    revisions_approved = sum(1 for item in item_details 
                           if item.get('revision_evaluation') and 
                           item.get('revision_evaluation', {}).get('approved'))
    revisions_rejected = revisions_requested - revisions_approved
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Plan Items", len(item_details))
    with col2:
        st.metric("Revisions Requested", revisions_requested)
    with col3:
        st.metric("Approved", revisions_approved)
    with col4:
        st.metric("To Implement", revisions_approved)
    
    # Show revision implementation preview
    if revisions_approved > 0:
        st.subheader("🔧 Revisions to Implement")
        st.success(f"Found {revisions_approved} approved revision(s) ready for implementation:")
        
        for i, item_detail in enumerate(item_details, 1):
            revision_evaluation = item_detail.get('revision_evaluation')
            if revision_evaluation and revision_evaluation.get('approved'):
                item_title = item_detail.get('item_title', f'Item {i}')
                revision_request = item_detail.get('revision_request', {})
                
                with st.expander(f"✅ {item_title} - Approved for Revision"):
                    st.write("**🔧 Revision to Implement:**")
                    st.info(revision_request.get('revision_request_content', 'N/A'))
                    
                    st.write("**🎯 Target Criteria:**")
                    targeted_criteria = revision_request.get('targeted_criteria', [])
                    for criterion in targeted_criteria:
                        st.write(f"  • {criterion}")
                    
                    st.write("**📊 Expected Impact:**")
                    impact = revision_evaluation.get('impact_assessment', 'N/A')
                    st.write(impact)
    else:
        st.info("🎉 No revisions approved for implementation - the plan is already well-optimized!")
        st.write("The plan will proceed as-is from Module 4.")
    
    # Show the goal and current plan
    if previous_output.get('goal'):
        st.info(f"**Goal:** {previous_output['goal']}")
    
    expanded_outline = previous_output.get('expanded_outline', {})
    if expanded_outline:
        st.subheader("📋 Current Plan")
        st.write(f"**{expanded_outline.get('plan_title', 'N/A')}**")
        st.write(f"*{expanded_outline.get('plan_description', 'N/A')}*")
        
        if expanded_outline.get('plan_items'):
            with st.expander("View Current Plan Items"):
                for i, item in enumerate(expanded_outline['plan_items'], 1):
                    st.write(f"**{i}. {item.get('item_title', 'N/A')}**")
                    description = item.get('item_description', 'N/A')
                    # Show truncated description
                    if len(description) > 300:
                        st.write(f"   {description[:300]}...")
                    else:
                        st.write(f"   {description}")
    
    with st.expander("View Full Module 4 Output JSON"):
        st.json(previous_output)
else:
    st.warning("⚠️ No output from Module 4 found. Please complete Module 4 first.")
    st.info("👈 Navigate to Module 4 using the sidebar to identify revisions first.")
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

# Input section for Module 5
st.markdown("---")
st.header("🔧 Configuration")

# Show what will happen
if revisions_approved > 0:
    st.info(f"""
    **Module 5 will:**
    1. 🔧 **Apply approved revisions** to {revisions_approved} plan item(s)
    2. 📝 **Generate revised text** incorporating the improvement suggestions
    3. ✅ **Evaluate implementation** to ensure revisions meet the targeted criteria
    4. 🔄 **Refine if needed** with multiple attempts to get the best results
    5. 📋 **Create final plan** with all improvements integrated
    6. 📊 **Generate summary** showing before/after criteria fulfillment
    """)
else:
    st.info("""
    **Module 5 will:**
    1. ✅ **Confirm no revisions needed** - plan is already optimized
    2. 📋 **Prepare final plan** using current expanded outline
    3. 📊 **Generate fulfillment summary** showing criteria achievement
    4. 🎯 **Ready for report generation** in Module 6
    """)

# Optional: Allow user to modify or confirm the input
confirm_input = st.checkbox("✅ Confirm input data and proceed with revision implementation", value=True)

if not confirm_input:
    st.info("Please confirm the input data above to proceed.")
    st.stop()

# Run button
st.markdown("---")
if st.button("🚀 Run Module 5: Implement Revisions", type="primary"):
    update_module_status('module5', 'in_progress')
    
    # Create placeholders
    status_placeholder = st.empty()
    log_container = st.container()
    
    # Debug container
    debug_container = st.expander("Debug Information", expanded=True)
    
    try:
        with st.spinner("Running Module 5..."):
            # Set the API key in environment
            os.environ['OPENAI_API_KEY'] = st.session_state.api_key
            
            with debug_container:
                st.write("Step 1: Environment setup")
                st.code(f"OPENAI_API_KEY set: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
                st.code(f"Module 4 output available: {'Yes' if previous_output else 'No'}")
                st.code(f"Revisions to implement: {revisions_approved}")
                
                # Try to import the module
                st.write("Step 2: Importing module5")
                try:
                    from module5 import run_module_5
                    st.success("✅ Successfully imported run_module_5")
                    st.code(f"run_module_5 type: {type(run_module_5)}")
                    st.code(f"run_module_5 module: {run_module_5.__module__}")
                except ImportError as e:
                    st.error(f"❌ Failed to import module5: {e}")
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
            
            status_placeholder.info("🔄 Initializing Module 5...")
            
            # Define the async wrapper function
            async def run_module_async():
                with debug_container:
                    st.write("Step 4: About to call run_module_5")
                    st.code(f"Input file: {input_file}")
                    st.code(f"Output file: {output_file}")
                
                # Call the actual function with captured output
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    result = await run_module_5(input_file, output_file)
                
                with debug_container:
                    st.write("Step 5: run_module_5 completed")
                    st.code(f"Return value: {result}")
                
                return result
            
            # Show running status with progress updates
            if revisions_approved > 0:
                progress_steps = [
                    f"🔧 Implementing {revisions_approved} approved revision(s)...",
                    "📝 Applying revisions to plan items...",
                    "✅ Evaluating implementation quality...",
                    "🔄 Refining implementations if needed...",
                    "📋 Creating final revised plan...",
                    "📊 Generating criteria fulfillment summary..."
                ]
            else:
                progress_steps = [
                    "✅ Confirming no revisions needed...",
                    "📋 Preparing final plan outline...",
                    "📊 Generating criteria fulfillment summary..."
                ]
            
            for i, step in enumerate(progress_steps):
                status_placeholder.info(step)
                if i < len(progress_steps) - 1:
                    # Short delay to show progress
                    import time
                    time.sleep(0.7)  # Slightly longer for Module 5
            
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
                        if 'revision_results' in output_data:
                            structure['revision_results'] = f"<list of {len(output_data['revision_results'])} results>"
                        st.json(structure)
                    
                    # Validate the output has expected structure
                    if not isinstance(output_data, dict):
                        raise Exception("Invalid output format: expected dictionary")
                    
                    # Save to session state
                    save_module_output('module5', output_data)
                    
                    # Get captured logs
                    stdout_log = stdout_capture.getvalue()
                    stderr_log = stderr_capture.getvalue()
                    
                    # Create standard log from captured output
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Extract key information
                    revision_results = output_data.get('revision_results', [])
                    revised_outline = output_data.get('revised_outline', {})
                    fulfillment_summary = output_data.get('criteria_fulfillment_summary', {})
                    
                    standard_log = f"""[{timestamp}] Module 5 Started
[{timestamp}] Input from Module 4: {revisions_approved} approved revisions
[{timestamp}] Running OpenAI Agents SDK...
[{timestamp}] Implementing approved revisions...
[{timestamp}] Processed {len(revision_results)} revision implementation(s)
[{timestamp}] Created final revised plan: {revised_outline.get('plan_title', 'N/A')}
[{timestamp}] Generated criteria fulfillment summary: {len(fulfillment_summary)} criteria
[{timestamp}] Module 5 Completed Successfully

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
                    
                    save_logs('module5', standard_log, verbose_log)
                    
                    # Clean up temp files
                    os.unlink(input_file)
                    os.unlink(output_file)
                    
                    status_placeholder.success("✅ Module 5 completed successfully!")
                    update_module_status('module5', 'completed')
                    
                    # Show brief summary
                    with log_container:
                        if revisions_approved > 0:
                            st.info(f"""
                            **Success!** Revision implementation complete:
                            - Implemented {len(revision_results)} revision(s)
                            - Created final revised plan with improvements
                            - Generated criteria fulfillment analysis
                            - Plan is now optimized and ready for final report
                            """)
                        else:
                            st.info(f"""
                            **Success!** Plan finalization complete:
                            - Confirmed no revisions needed
                            - Prepared final optimized plan
                            - Generated criteria fulfillment analysis
                            - Plan is ready for final report generation
                            """)
                else:
                    raise Exception("Output file is empty - module may have failed silently")
                
            else:
                with debug_container:
                    st.error("Output file not created!")
                    st.code(f"File exists: {os.path.exists(output_file)}")
                raise Exception("Output file not created - module may have failed")
            
    except ImportError as e:
        update_module_status('module5', 'failed')
        st.error(f"❌ Error importing module5: {str(e)}")
        st.error("Make sure module5.py is in the parent directory of streamlit_app/")
        
    except Exception as e:
        update_module_status('module5', 'failed')
        st.error(f"❌ Error running Module 5: {str(e)}")
        st.code(traceback.format_exc())
        
        # Show any captured output for debugging
        if 'stdout_capture' in locals() and stdout_capture.getvalue():
            st.text("Captured output:")
            st.code(stdout_capture.getvalue())
        if 'stderr_capture' in locals() and stderr_capture.getvalue():
            st.text("Captured errors:")
            st.code(stderr_capture.getvalue())

# Output section
if st.session_state.module_outputs.get('module5'):
    st.markdown("---")
    st.header("📤 Module 5 Output")
    
    output_data = st.session_state.module_outputs['module5']
    
    # Display key metrics
    revision_results = output_data.get('revision_results', [])
    revised_outline = output_data.get('revised_outline', {})
    fulfillment_summary = output_data.get('criteria_fulfillment_summary', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Revisions Implemented", len(revision_results))
    with col2:
        if revised_outline:
            revised_items = len(revised_outline.get('plan_items', []))
            st.metric("Final Plan Items", revised_items)
        else:
            st.metric("Final Plan Items", 0)
    with col3:
        if fulfillment_summary:
            total_improvements = sum(
                stats.get('fully_met_revisions', 0) + stats.get('partially_met_revisions', 0)
                for stats in fulfillment_summary.values()
            )
            st.metric("Criteria Improvements", total_improvements)
        else:
            st.metric("Criteria Improvements", 0)
    with col4:
        # Calculate success rate from fulfillment summary
        if fulfillment_summary:
            total_attempts = sum(
                stats.get('fully_met_revisions', 0) + 
                stats.get('partially_met_revisions', 0) + 
                stats.get('not_met_revisions', 0)
                for stats in fulfillment_summary.values()
            )
            successful = sum(
                stats.get('fully_met_revisions', 0)
                for stats in fulfillment_summary.values()
            )
            success_rate = (successful / total_attempts * 100) if total_attempts > 0 else 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("Success Rate", "100%")
    
    # Show criteria fulfillment summary
    if fulfillment_summary:
        st.subheader("📊 Criteria Fulfillment Summary")
        
        for criterion_name, stats in fulfillment_summary.items():
            with st.expander(f"🎯 {criterion_name}"):
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.write("**Original Results:**")
                    st.write(f"✅ Pass: {stats.get('original_pass', 0)}")
                    st.write(f"❌ Fail: {stats.get('original_fail', 0)}")
                
                with col_b:
                    st.write("**After Revisions:**")
                    fully_met = stats.get('fully_met_revisions', 0)
                    partially_met = stats.get('partially_met_revisions', 0)
                    not_met = stats.get('not_met_revisions', 0)
                    
                    if fully_met > 0:
                        st.success(f"🎯 Fully Met: {fully_met}")
                    if partially_met > 0:
                        st.warning(f"🔄 Partially Met: {partially_met}")
                    if not_met > 0:
                        st.error(f"❌ Not Met: {not_met}")
                    if fully_met == 0 and partially_met == 0 and not_met == 0:
                        st.info("No revisions for this criterion")
                
                with col_c:
                    original_fail = stats.get('original_fail', 0)
                    if original_fail > 0 and fully_met > 0:
                        improvement_rate = (fully_met / original_fail) * 100
                        st.write("**Improvement:**")
                        st.success(f"📈 {improvement_rate:.1f}% of failures resolved")
                    elif original_fail == 0:
                        st.write("**Status:**")
                        st.success("✅ Already optimal")
                    else:
                        st.write("**Status:**")
                        st.info("No improvements applied")
    
    # Show revision implementation results
    if revision_results:
        st.subheader("🔧 Revision Implementation Results")
        
        for i, result in enumerate(revision_results, 1):
            item_title = result.get('item_title', f'Item {i}')
            attempt_count = result.get('attempt_count', 1)
            implementation_eval = result.get('implementation_evaluation', {})
            meets_criteria = implementation_eval.get('meets_criteria', False)
            
            status_emoji = "✅" if meets_criteria else "⚠️"
            status_text = "SUCCESSFUL" if meets_criteria else "PARTIAL"
            
            with st.expander(f"{status_emoji} {item_title} - {status_text} ({attempt_count} attempt(s))"):
                
                # Show revision request
                revision_request = result.get('revision_request', {})
                if revision_request:
                    st.write("**🔧 Original Revision Request:**")
                    st.info(revision_request.get('revision_request_content', 'N/A'))
                
                # Show implementation evaluation
                if implementation_eval:
                    st.write("**✅ Implementation Evaluation:**")
                    if meets_criteria:
                        st.success("✅ Successfully meets targeted criteria")
                    else:
                        st.warning("⚠️ Partially meets criteria")
                    
                    st.write("**📊 Reasoning:**")
                    st.write(implementation_eval.get('reasoning', 'N/A'))
                    
                    st.write("**🎯 Criteria Fulfillment:**")
                    criteria_fulfillment = implementation_eval.get('criteria_fulfillment', 'N/A')
                    st.write(criteria_fulfillment)
                    
                    if implementation_eval.get('improvement_suggestions'):
                        st.write("**💡 Improvement Suggestions:**")
                        st.write(implementation_eval.get('improvement_suggestions'))
                
                # Show final text preview
                final_text = result.get('final_text', '')
                if final_text:
                    st.write("**📝 Final Revised Text (Preview):**")
                    if len(final_text) > 500:
                        st.write(f"{final_text[:500]}...")
                        if st.button(f"Show full text for {item_title}", key=f"show_full_{i}"):
                            st.text_area("Full revised text:", final_text, height=300, key=f"full_text_{i}")
                    else:
                        st.write(final_text)
    
    # Show final revised plan
    if revised_outline:
        st.subheader("📋 Final Revised Plan")
        
        st.success(f"**{revised_outline.get('plan_title', 'N/A')}**")
        st.write(f"**Description:** {revised_outline.get('plan_description', 'N/A')}")
        st.write(f"**Rating:** {revised_outline.get('rating', 'N/A')}/10")
        st.write(f"**Created by:** {revised_outline.get('created_by', 'N/A')}")
        
        if revised_outline.get('plan_items'):
            st.write("**Final Plan Items:**")
            for i, item in enumerate(revised_outline['plan_items'], 1):
                with st.expander(f"📋 Step {i}: {item.get('item_title', 'N/A')}"):
                    description = item.get('item_description', 'N/A')
                    # Show description with smart truncation
                    if len(description) > 1000:
                        st.write(f"{description[:1000]}...")
                        if st.button(f"Show full description for Step {i}", key=f"expand_final_{i}"):
                            st.text_area("Full description:", description, height=400, key=f"full_desc_{i}")
                    else:
                        st.write(description)
    
    # Display the full output JSON
    st.subheader("📄 Full Output JSON")
    display_json(output_data)
    
    # Download options
    st.subheader("📥 Downloads")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        download_json(output_data, "module5_output.json")
    
    logs = st.session_state.current_logs.get('module5', {})
    with col2:
        if logs.get('standard'):
            download_text(logs['standard'], "module5_standard.log", "📥 Download Standard Log")
    
    with col3:
        if logs.get('verbose'):
            download_text(logs['verbose'], "module5_verbose.log", "📥 Download Verbose Log")
    
    # Send to next module button
    st.markdown("---")
    st.success("🎉 Plan optimization complete! Your refined plan is ready for final report generation.")
    
    if st.button("📨 Send to Module 6", type="primary"):
        st.success("✅ Output ready for Module 6!")
        st.info("Navigate to Module 6 using the sidebar to generate your final markdown report.")

# Display logs if available
if st.session_state.current_logs.get('module5'):
    st.markdown("---")
    st.header("📋 Logs")
    
    logs = st.session_state.current_logs['module5']
    log_type = st.radio("Select log type:", ["Standard", "Verbose"], key="module5_log_type")
    
    if log_type == "Standard" and logs.get('standard'):
        st.text_area("Standard Log", value=logs['standard'], height=300, disabled=True, key="module5_standard_log")
    elif log_type == "Verbose" and logs.get('verbose'):
        st.text_area("Verbose Log", value=logs['verbose'], height=300, disabled=True, key="module5_verbose_log")

# Sidebar status
st.sidebar.header("Module 5 Status")
status = st.session_state.module_status['module5']
status_emoji = {
    'not_started': '⭕',
    'in_progress': '🔄',
    'completed': '✅',
    'failed': '❌'
}.get(status, '❓')
st.sidebar.info(f"Status: {status_emoji} {status.replace('_', ' ').title()}")

if output_data := st.session_state.module_outputs.get('module5'):
    revision_results = output_data.get('revision_results', [])
    revised_outline = output_data.get('revised_outline', {})
    
    if len(revision_results) > 0:
        successful_revisions = sum(1 for result in revision_results 
                                 if result.get('implementation_evaluation', {}).get('meets_criteria'))
        st.sidebar.success(f"✅ {successful_revisions}/{len(revision_results)} revisions successful")
    else:
        st.sidebar.info("ℹ️ No revisions implemented")
    
    if revised_outline:
        st.sidebar.success(f"📋 Final plan ready")
        
        # Show quick improvement metrics
        fulfillment_summary = output_data.get('criteria_fulfillment_summary', {})
        if fulfillment_summary:
            total_improvements = sum(
                stats.get('fully_met_revisions', 0)
                for stats in fulfillment_summary.values()
            )
            if total_improvements > 0:
                st.sidebar.metric("Improvements", f"+{total_improvements}")
