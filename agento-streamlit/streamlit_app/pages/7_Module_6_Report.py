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

# Check if module6.py exists
module6_path = os.path.join(module_dir, 'module6.py')
st.sidebar.write(f"module6.py exists: {os.path.exists(module6_path)}")
if os.path.exists(module6_path):
    st.sidebar.write(f"module6.py size: {os.path.getsize(module6_path)} bytes")

from utils.session_state import init_session_state, save_module_output, update_module_status, get_previous_module_output, save_logs
from utils.file_handlers import download_json, download_text, display_json

# Initialize session state
init_session_state()

st.title("📄 Module 6: Report Generation")
st.markdown("Generate a final markdown report from your refined plan.")

# Check API key
if not st.session_state.api_key:
    st.error("❌ Please configure your OpenAI API key in the API Configuration page first.")
    st.stop()

# Debug: Show API key status
st.sidebar.write(f"API Key set: {'Yes' if st.session_state.api_key else 'No'}")
if st.session_state.api_key:
    st.sidebar.write(f"API Key prefix: {st.session_state.api_key[:10]}...")

# Check for previous module output
previous_output = get_previous_module_output('module6')
if previous_output:
    st.success("✅ Input available from Module 5")
    
    # Display summary of Module 5 output
    st.subheader("📥 Input from Module 5")
    
    # Get key information
    revised_outline = previous_output.get('revised_outline', {})
    revision_results = previous_output.get('revision_results', [])
    fulfillment_summary = previous_output.get('criteria_fulfillment_summary', {})
    selected_criteria = previous_output.get('selected_criteria', [])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if revised_outline:
            plan_items = len(revised_outline.get('plan_items', []))
            st.metric("Final Plan Items", plan_items)
        else:
            st.metric("Final Plan Items", 0)
    with col2:
        st.metric("Revisions Applied", len(revision_results))
    with col3:
        st.metric("Success Criteria", len(selected_criteria))
    with col4:
        if fulfillment_summary:
            total_improvements = sum(
                stats.get('fully_met_revisions', 0)
                for stats in fulfillment_summary.values()
            )
            st.metric("Improvements Made", total_improvements)
        else:
            st.metric("Improvements Made", 0)
    
    # Show the final plan preview
    if revised_outline:
        st.subheader("📋 Final Plan Preview")
        st.success(f"**{revised_outline.get('plan_title', 'N/A')}**")
        st.write(f"*{revised_outline.get('plan_description', 'N/A')}*")
        
        if revised_outline.get('plan_items'):
            st.write(f"**Plan contains {len(revised_outline['plan_items'])} detailed steps**")
            
            with st.expander("Preview Plan Steps"):
                for i, item in enumerate(revised_outline['plan_items'], 1):
                    st.write(f"**{i}. {item.get('item_title', 'N/A')}**")
                    description = item.get('item_description', 'N/A')
                    # Show brief preview
                    if len(description) > 150:
                        st.write(f"   {description[:150]}...")
                    else:
                        st.write(f"   {description}")
    
    # Show the goal
    if previous_output.get('goal'):
        st.info(f"**Goal:** {previous_output['goal']}")
    
    # Show success criteria
    if selected_criteria:
        with st.expander("View Success Criteria"):
            for i, criterion in enumerate(selected_criteria, 1):
                if isinstance(criterion, dict):
                    st.write(f"**{i}.** {criterion.get('criteria', 'N/A')}")
                    st.caption(f"   *{criterion.get('reasoning', 'N/A')}*")
                else:
                    st.write(f"**{i}.** {criterion}")
    
    with st.expander("View Full Module 5 Output JSON"):
        st.json(previous_output)
else:
    st.warning("⚠️ No output from Module 5 found. Please complete Module 5 first.")
    st.info("👈 Navigate to Module 5 using the sidebar to implement revisions first.")
    st.stop()

# Simple function to run module6 (it's not async)
def run_module6_sync(input_data):
    """Run module6 synchronously since it just formats markdown"""
    try:
        # Import module6 which has the main() function
        import importlib.util
        
        # Create a temporary input file for module6
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_input_file:
            input_file = tmp_input_file.name
            json.dump(input_data, tmp_input_file, indent=2)
        
        # Create a temporary output file
        output_file = tempfile.mktemp(suffix='.md')
        
        # Module6 works with files, so we'll simulate its main function
        # But first let's import it properly
        from module6 import main as module6_main
        
        # Temporarily replace sys.argv to pass our file paths
        original_argv = sys.argv
        try:
            # Module6 expects to be called directly, so we need to simulate that
            # Let's read the module6.py code and adapt it
            
            # Read input data
            goal = input_data.get("goal", "No goal provided.")
            selected_criteria = input_data.get("selected_criteria", [])
            revised_outline = input_data.get("revised_outline", {})
            
            # Build the markdown content (adapted from module6.py logic)
            lines = []
            
            # Title 
            plan_title = revised_outline.get("plan_title", "No Plan Title")
            lines.append("# " + plan_title)
            lines.append("")
            
            # Goal
            lines.append("## Goal")
            lines.append("")
            lines.append(goal)
            lines.append("")
            
            # Success Criteria
            lines.append("## Success Criteria")
            lines.append("")
            if not selected_criteria:
                lines.append("No success criteria provided.")
            else:
                for criterion in selected_criteria:
                    if isinstance(criterion, dict):
                        criteria_text = criterion.get("criteria", "")
                        if criteria_text:
                            lines.append(f"- **{criteria_text}**")
                    else:
                        lines.append(f"- **{criterion}**")
            lines.append("")
            
            # Plan overview
            plan_description = revised_outline.get("plan_description", "")
            if plan_description:
                lines.append("## Plan Overview")
                lines.append("")
                lines.append(plan_description)
                lines.append("")
            
            # Detailed steps
            lines.append("## Detailed Implementation Steps")
            lines.append("")
            
            plan_items = revised_outline.get("plan_items", [])
            for i, item in enumerate(plan_items, 1):
                item_title = item.get("item_title", f"Step {i}")
                item_description = item.get("item_description", "")
                
                # Add the step title
                lines.append(f"### Step {i}: {item_title}")
                lines.append("")
                
                # Add the description (clean up any markdown formatting issues)
                cleaned_content = item_description.replace("### Step", "#### Step")  # Prevent header conflicts
                lines.append(cleaned_content)
                lines.append("")
            
            # Add performance summary if available
            fulfillment_summary = input_data.get('criteria_fulfillment_summary', {})
            if fulfillment_summary:
                lines.append("## Performance Summary")
                lines.append("")
                lines.append("This plan has been optimized based on the success criteria:")
                lines.append("")
                
                for criterion_name, stats in fulfillment_summary.items():
                    original_fail = stats.get('original_fail', 0)
                    fully_met = stats.get('fully_met_revisions', 0)
                    
                    if original_fail > 0 and fully_met > 0:
                        improvement_rate = (fully_met / original_fail) * 100
                        lines.append(f"- **{criterion_name}**: {improvement_rate:.0f}% improvement through revisions")
                    elif original_fail == 0:
                        lines.append(f"- **{criterion_name}**: Already optimized")
                    else:
                        lines.append(f"- **{criterion_name}**: Evaluated and confirmed")
                
                lines.append("")
            
            # Add closing note
            lines.append("---")
            lines.append("")
            lines.append("*This plan was generated using Dazza Greenwood's Agento framework.*")
            lines.append("")
            lines.append(f"*Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*")
            
            content = "\n".join(lines)
            
            # Write to output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Read back the content
            with open(output_file, 'r', encoding='utf-8') as f:
                final_content = f.read()
            
            # Clean up temp files
            os.unlink(input_file)
            os.unlink(output_file)
            
            return final_content
            
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        st.error(f"Error in module6 processing: {str(e)}")
        return None

# Input section for Module 6
st.markdown("---")
st.header("🔧 Configuration")

# Show what will happen
st.info("""
**Module 6 will:**
1. 📝 **Generate markdown report** from your final refined plan
2. 🎯 **Include goal and success criteria** for context
3. 📋 **Format detailed implementation steps** in a readable structure
4. 📊 **Add performance summary** showing improvements made
5. 📄 **Create downloadable .md file** for easy sharing
6. 🖥️ **Display formatted report** on screen for immediate review
""")

# Optional: Allow user to modify or confirm the input
confirm_input = st.checkbox("✅ Confirm input data and proceed with report generation", value=True)

if not confirm_input:
    st.info("Please confirm the input data above to proceed.")
    st.stop()

# Run button
st.markdown("---")
if st.button("🚀 Run Module 6: Generate Report", type="primary"):
    update_module_status('module6', 'in_progress')
    
    # Create placeholders
    status_placeholder = st.empty()
    log_container = st.container()
    
    # Debug container
    debug_container = st.expander("Debug Information", expanded=True)
    
    try:
        with st.spinner("Running Module 6..."):
            
            with debug_container:
                st.write("Step 1: Environment setup")
                st.code(f"Module 5 output available: {'Yes' if previous_output else 'No'}")
                
                # Try to import the module
                st.write("Step 2: Importing module6")
                try:
                    import module6
                    st.success("✅ Successfully imported module6")
                    st.code(f"module6 type: {type(module6)}")
                except ImportError as e:
                    st.error(f"❌ Failed to import module6: {e}")
                    st.code(traceback.format_exc())
                    raise
            
            status_placeholder.info("🔄 Generating markdown report...")
            
            # Show running status with progress updates
            progress_steps = [
                "📝 Formatting goal and success criteria...",
                "📋 Structuring plan implementation steps...",
                "📊 Adding performance summary...",
                "🎨 Generating final markdown format...",
                "✅ Preparing report for display and download..."
            ]
            
            for i, step in enumerate(progress_steps):
                status_placeholder.info(step)
                # Short delay to show progress
                import time
                time.sleep(0.3)
            
            with debug_container:
                st.write("Step 3: Running module6 markdown generation")
            
            # Generate the markdown report
            markdown_content = run_module6_sync(previous_output)
            
            with debug_container:
                st.write("Step 4: Module6 completed")
                if markdown_content:
                    st.code(f"Generated markdown length: {len(markdown_content)} characters")
                    st.text_area("Generated markdown (first 500 chars):", markdown_content[:500], height=150)
                else:
                    st.error("No markdown content generated")
            
            if markdown_content:
                # Save to session state
                module6_output = {
                    "goal": previous_output.get("goal", ""),
                    "selected_criteria": previous_output.get("selected_criteria", []),
                    "revised_outline": previous_output.get("revised_outline", {}),
                    "markdown_report": markdown_content,
                    "generated_at": datetime.now().isoformat(),
                    "report_stats": {
                        "character_count": len(markdown_content),
                        "line_count": len(markdown_content.split('\n')),
                        "word_count": len(markdown_content.split())
                    }
                }
                
                save_module_output('module6', module6_output)
                
                # Create logs
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                standard_log = f"""[{timestamp}] Module 6 Started
[{timestamp}] Input from Module 5: Final plan with {len(previous_output.get('revised_outline', {}).get('plan_items', []))} items
[{timestamp}] Generating markdown report...
[{timestamp}] Report generated: {len(markdown_content)} characters, {len(markdown_content.split())} words
[{timestamp}] Module 6 Completed Successfully
""".strip()
                
                verbose_log = f"""{standard_log}

--- Generated Markdown Report ---
{markdown_content}
""".strip()
                
                save_logs('module6', standard_log, verbose_log)
                
                status_placeholder.success("✅ Module 6 completed successfully!")
                update_module_status('module6', 'completed')
                
                # Show brief summary
                with log_container:
                    st.info(f"""
                    **Success!** Markdown report generated:
                    - {len(markdown_content.split())} words
                    - {len(markdown_content.split('\n'))} lines
                    - Ready for download and sharing
                    """)
            else:
                raise Exception("Failed to generate markdown content")
            
    except ImportError as e:
        update_module_status('module6', 'failed')
        st.error(f"❌ Error importing module6: {str(e)}")
        st.error("Make sure module6.py is in the parent directory of streamlit_app/")
        
    except Exception as e:
        update_module_status('module6', 'failed')
        st.error(f"❌ Error running Module 6: {str(e)}")
        st.code(traceback.format_exc())

# Output section
if st.session_state.module_outputs.get('module6'):
    st.markdown("---")
    st.header("📤 Module 6 Output")
    
    output_data = st.session_state.module_outputs['module6']
    markdown_content = output_data.get('markdown_report', '')
    report_stats = output_data.get('report_stats', {})
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Word Count", report_stats.get('word_count', 0))
    with col2:
        st.metric("Lines", report_stats.get('line_count', 0))
    with col3:
        st.metric("Characters", report_stats.get('character_count', 0))
    with col4:
        generated_at = output_data.get('generated_at', '')
        if generated_at:
            try:
                gen_time = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
                st.metric("Generated", gen_time.strftime('%m/%d %I:%M%p'))
            except:
                st.metric("Generated", "Just now")
        else:
            st.metric("Generated", "Just now")
    
    # Display the markdown report
    if markdown_content:
        st.subheader("📄 Final Report")
        
        # Add some styling for the report display
        st.markdown("""
        <style>
        .report-container {
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            background-color: #fafafa;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display the rendered markdown
        with st.container():
            st.markdown('<div class="report-container">', unsafe_allow_html=True)
            st.markdown(markdown_content)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Raw markdown view
        with st.expander("📝 View Raw Markdown"):
            st.code(markdown_content, language='markdown')
    
    # Download options
    st.subheader("📥 Downloads")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if markdown_content:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plan_title = output_data.get('revised_outline', {}).get('plan_title', 'plan')
            # Clean filename
            clean_title = "".join(c for c in plan_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            clean_title = clean_title.replace(' ', '_')[:30]  # Limit length
            filename = f"{clean_title}_{timestamp}.md"
            
            st.download_button(
                label="📄 Download Markdown Report",
                data=markdown_content,
                file_name=filename,
                mime='text/markdown',
                key="download_markdown"
            )
    
    with col2:
        download_json(output_data, "module6_output.json")
    
    logs = st.session_state.current_logs.get('module6', {})
    with col3:
        if logs.get('standard'):
            download_text(logs['standard'], "module6_standard.log", "📥 Download Standard Log")
    
    with col4:
        if logs.get('verbose'):
            download_text(logs['verbose'], "module6_verbose.log", "📥 Download Verbose Log")
    
    # Completion celebration
    st.markdown("---")
    st.success("🎉 **Congratulations!** Your Agento planning process is complete!")
    
    st.info("""
    **What you've accomplished:**
    - ✅ Generated success criteria for your goal
    - ✅ Created and evaluated multiple plan outlines  
    - ✅ Expanded plan items into detailed descriptions
    - ✅ Identified and implemented improvements
    - ✅ Generated a comprehensive final report
    
    **Your plan is now ready to implement!**
    """)
    
    # Option to restart
    if st.button("🔄 Start New Planning Process", type="secondary"):
        # Clear all session state
        for key in ['module_outputs', 'module_status', 'current_logs']:
            if key in st.session_state:
                if key == 'module_status':
                    st.session_state[key] = {f'module{i}': 'not_started' for i in range(1, 7)}
                else:
                    st.session_state[key] = {}
        st.success("✅ Session cleared! Navigate to Module 1 to start a new planning process.")
        st.rerun()

# Display logs if available
if st.session_state.current_logs.get('module6'):
    st.markdown("---")
    st.header("📋 Logs")
    
    logs = st.session_state.current_logs['module6']
    log_type = st.radio("Select log type:", ["Standard", "Verbose"], key="module6_log_type")
    
    if log_type == "Standard" and logs.get('standard'):
        st.text_area("Standard Log", value=logs['standard'], height=300, disabled=True, key="module6_standard_log")
    elif log_type == "Verbose" and logs.get('verbose'):
        st.text_area("Verbose Log", value=logs['verbose'], height=300, disabled=True, key="module6_verbose_log")

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

if output_data := st.session_state.module_outputs.get('module6'):
    report_stats = output_data.get('report_stats', {})
    
    st.sidebar.success("✅ Report generated")
    
    if report_stats:
        st.sidebar.metric("Words", report_stats.get('word_count', 0))
        st.sidebar.metric("Lines", report_stats.get('line_count', 0))
    
    # Show completion status
    st.sidebar.markdown("---")
    st.sidebar.success("🎉 **Process Complete!**")
    st.sidebar.info("All 6 modules finished successfully")
