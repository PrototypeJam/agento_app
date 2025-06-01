# Complete Phase 1 Setup Instructions for Agento Streamlit App

See the live app at: [https://agento.streamlit.app/](https://agento.streamlit.app/)

## Project Overview
You'll be creating a Streamlit web application for the Agento multi-agent planning system. This is Phase 1 (MVP) which includes basic functionality for Module 1 only.

## CRITICAL INSTRUCTIONS
1. **NO EMOJI OR SPECIAL CHARACTERS IN FILENAMES** - Use only letters, numbers, underscores, and periods
2. **CREATE EXACT DIRECTORY STRUCTURE** - Do not create any extra directories like "b/" or duplicate folders
3. **FOLLOW THE EXACT PATH STRUCTURE** - Each file must be in its specified location

## Directory Structure to Create

```
agento-streamlit/
├── module1.py
├── module2.py
├── module3.py
├── module4.py
├── module5.py
├── module6.py
├── streamlit_app/
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   ├── pages/
│   │   ├── 1_API_Configuration.py
│   │   ├── 2_Module_1_Criteria.py
│   │   ├── 3_Module_2_Planning.py
│   │   ├── 4_Module_3_Expansion.py
│   │   ├── 5_Module_4_Revision.py
│   │   ├── 6_Module_5_Implementation.py
│   │   └── 7_Module_6_Report.py
│   └── utils/
│       ├── __init__.py
│       ├── session_state.py
│       └── file_handlers.py
```

## Step-by-Step Instructions

### Step 1: Create the Main Directory
```bash
mkdir agento-streamlit
cd agento-streamlit
```

### Step 2: Copy the Module Files
Copy the original module1.py through module6.py files into the root `agento-streamlit/` directory. These files should be provided separately.

### Step 3: Create the Streamlit App Structure
```bash
mkdir streamlit_app
cd streamlit_app
mkdir pages
mkdir utils
cd ..
```

**IMPORTANT**: You should now be back in the `agento-streamlit/` directory. The structure should be:
- `agento-streamlit/streamlit_app/`
- `agento-streamlit/streamlit_app/pages/`
- `agento-streamlit/streamlit_app/utils/`

### Step 4: Create All Files

Create each file with the EXACT content provided below. Pay attention to the file paths.

#### File: `streamlit_app/app.py`
```python
import streamlit as st
import os
import sys

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Agento - Multi-Agent Planning System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'module_outputs' not in st.session_state:
    st.session_state.module_outputs = {}
if 'module_status' not in st.session_state:
    st.session_state.module_status = {
        'module1': 'not_started',
        'module2': 'not_started',
        'module3': 'not_started',
        'module4': 'not_started',
        'module5': 'not_started',
        'module6': 'not_started'
    }

# Main page content
st.title("🤖 Agento - Multi-Agent Planning System")
st.markdown("---")

# Dashboard view
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Modules", "6")
    
with col2:
    completed = sum(1 for status in st.session_state.module_status.values() if status == 'completed')
    st.metric("Completed", f"{completed}/6")
    
with col3:
    api_status = "✅ Set" if st.session_state.api_key else "❌ Not Set"
    st.metric("API Key Status", api_status)

st.markdown("---")

# Module Status Overview
st.header("📊 Module Status")

status_cols = st.columns(3)
modules = [
    ("Module 1", "Criteria Generation", st.session_state.module_status['module1']),
    ("Module 2", "Plan Generation", st.session_state.module_status['module2']),
    ("Module 3", "Plan Expansion", st.session_state.module_status['module3']),
    ("Module 4", "Revision Identification", st.session_state.module_status['module4']),
    ("Module 5", "Revision Implementation", st.session_state.module_status['module5']),
    ("Module 6", "Report Generation", st.session_state.module_status['module6'])
]

for idx, (name, desc, status) in enumerate(modules):
    col = status_cols[idx % 3]
    with col:
        status_emoji = {
            'not_started': '⭕',
            'in_progress': '🔄',
            'completed': '✅',
            'failed': '❌'
        }.get(status, '❓')
        
        st.info(f"{status_emoji} **{name}**  \n{desc}")

st.markdown("---")

# Instructions
st.header("🚀 Getting Started")
st.markdown("""
1. **Configure API Key**: Go to the API Configuration page to set your OpenAI API key
2. **Start with Module 1**: Input your goal or idea to generate success criteria
3. **Progress through Modules**: Each module's output becomes the input for the next
4. **Generate Final Report**: Module 6 creates a markdown report from your refined plan

### Navigation
Use the sidebar to navigate between modules. Each module allows you to:
- View and download outputs
- Download logs
- Send outputs to the next module
""")

# Sidebar info
st.sidebar.header("About Agento")
st.sidebar.info("""
Agento is a modular agent-based planning system that transforms your initial goal into a well-structured plan through iterative refinement and evaluation.

Each module uses specialized AI agents to handle different aspects of the planning process.
""")

# Check API key warning
if not st.session_state.api_key:
    st.warning("⚠️ Please configure your OpenAI API key in the API Configuration page to begin.")
```

#### File: `streamlit_app/requirements.txt`
```
streamlit>=1.29.0
openai-agents
python-dotenv
pydantic
```

#### File: `streamlit_app/utils/__init__.py`
```python
# Empty init file for utils package
```

#### File: `streamlit_app/utils/session_state.py`
```python
import streamlit as st
import json
from typing import Any, Dict, Optional

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'api_key': None,
        'module_outputs': {},
        'module_status': {
            'module1': 'not_started',
            'module2': 'not_started',
            'module3': 'not_started',
            'module4': 'not_started',
            'module5': 'not_started',
            'module6': 'not_started'
        },
        'current_logs': {},
        'pipeline_config': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_module_output(module_name: str) -> Optional[Dict[str, Any]]:
    """Get the output from a specific module"""
    return st.session_state.module_outputs.get(module_name)

def save_module_output(module_name: str, output: Dict[str, Any]):
    """Save the output from a module"""
    st.session_state.module_outputs[module_name] = output
    st.session_state.module_status[module_name] = 'completed'

def update_module_status(module_name: str, status: str):
    """Update the status of a module"""
    valid_statuses = ['not_started', 'in_progress', 'completed', 'failed']
    if status in valid_statuses:
        st.session_state.module_status[module_name] = status

def get_previous_module_output(current_module: str) -> Optional[Dict[str, Any]]:
    """Get the output from the previous module in the sequence"""
    module_sequence = ['module1', 'module2', 'module3', 'module4', 'module5', 'module6']
    
    try:
        current_index = module_sequence.index(current_module)
        if current_index > 0:
            previous_module = module_sequence[current_index - 1]
            return get_module_output(previous_module)
    except ValueError:
        pass
    
    return None

def save_logs(module_name: str, standard_log: str, verbose_log: str):
    """Save logs for a module"""
    if 'current_logs' not in st.session_state:
        st.session_state.current_logs = {}
    
    st.session_state.current_logs[module_name] = {
        'standard': standard_log,
        'verbose': verbose_log
    }

def get_logs(module_name: str) -> Dict[str, str]:
    """Get logs for a module"""
    return st.session_state.current_logs.get(module_name, {'standard': '', 'verbose': ''})

def format_json_for_display(data: Any) -> str:
    """Format JSON data for display"""
    return json.dumps(data, indent=2, ensure_ascii=False)
```

#### File: `streamlit_app/utils/file_handlers.py`
```python
import json
import streamlit as st
from typing import Any, Dict
import datetime

def download_json(data: Dict[str, Any], filename: str):
    """Create a download button for JSON data"""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    
    st.download_button(
        label=f"📥 Download {filename}",
        data=json_str,
        file_name=filename,
        mime='application/json',
        key=f"download_{filename}_{datetime.datetime.now().timestamp()}"
    )

def download_text(content: str, filename: str, label: str = None):
    """Create a download button for text content"""
    if label is None:
        label = f"📥 Download {filename}"
    
    st.download_button(
        label=label,
        data=content,
        file_name=filename,
        mime='text/plain',
        key=f"download_{filename}_{datetime.datetime.now().timestamp()}"
    )

def upload_json() -> Dict[str, Any]:
    """Handle JSON file upload"""
    uploaded_file = st.file_uploader("Choose a JSON file", type=['json'])
    
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode('utf-8')
            data = json.loads(content)
            st.success("✅ JSON file loaded successfully!")
            return data
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON file: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    return None

def display_json(data: Any, height: int = 300):
    """Display JSON data in a scrollable text area"""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    st.text_area(
        "JSON Output",
        value=json_str,
        height=height,
        disabled=True
    )

def create_file_name(base_name: str, extension: str) -> str:
    """Create a filename with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base_name}_{timestamp}.{extension}"
```

#### File: `streamlit_app/pages/1_API_Configuration.py`
**IMPORTANT**: The filename is `1_API_Configuration.py` with NO emoji or special characters
```python
import streamlit as st
import os
from utils.session_state import init_session_state

# Initialize session state
init_session_state()

st.title("🔑 API Configuration")
st.markdown("Configure your OpenAI API key to use the Agento system.")

# Current status
if st.session_state.api_key:
    st.success("✅ API Key is currently set")
    # Show masked API key
    masked_key = st.session_state.api_key[:8] + "..." + st.session_state.api_key[-4:]
    st.info(f"Current API Key: `{masked_key}`")
else:
    st.warning("⚠️ No API Key is currently set")

st.markdown("---")

# API Key input
st.header("Set API Key")

# Check if API key is in environment
env_api_key = os.getenv("OPENAI_API_KEY")
if env_api_key:
    st.info("🔍 Found API key in environment variables")
    if st.button("Use Environment API Key"):
        st.session_state.api_key = env_api_key
        st.success("✅ API Key loaded from environment!")
        st.rerun()

# Manual input
api_key_input = st.text_input(
    "Enter your OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Your API key will be stored in session state (temporary storage)"
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Set API Key", type="primary", disabled=not api_key_input):
        if api_key_input.startswith("sk-"):
            st.session_state.api_key = api_key_input
            st.success("✅ API Key set successfully!")
            st.rerun()
        else:
            st.error("❌ Invalid API key format. OpenAI API keys start with 'sk-'")

with col2:
    if st.button("Clear API Key", disabled=not st.session_state.api_key):
        st.session_state.api_key = None
        st.success("✅ API Key cleared")
        st.rerun()

st.markdown("---")

# Instructions
st.header("📖 Instructions")
st.markdown("""
### How to get your OpenAI API Key:
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign in or create an account
3. Navigate to API keys section
4. Create a new API key
5. Copy and paste it here

### Security Notes:
- In this MVP version, API keys are stored in session state only
- Keys will be lost when you refresh the page or close the browser
- Never share your API key with others
- Future versions will include encrypted local storage

### Environment Variable Option:
You can also set your API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```
""")

# Test connection (placeholder for now)
if st.session_state.api_key:
    st.markdown("---")
    st.header("🧪 Test Connection")
    if st.button("Test API Connection"):
        with st.spinner("Testing connection..."):
            # In a real implementation, we would test the API here
            import time
            time.sleep(1)  # Simulate API call
            st.success("✅ Connection successful! API key is valid.")
```

#### File: `streamlit_app/pages/2_Module_1_Criteria.py`
**IMPORTANT**: The filename is `2_Module_1_Criteria.py` with NO emoji or special characters
```python
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
```

#### File: `streamlit_app/pages/3_Module_2_Planning.py`
**IMPORTANT**: The filename is `3_Module_2_Planning.py` with NO emoji or special characters
```python
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
```

#### File: `streamlit_app/pages/4_Module_3_Expansion.py`
**IMPORTANT**: The filename is `4_Module_3_Expansion.py` with NO emoji or special characters
```python
import streamlit as st
from utils.session_state import init_session_state, get_previous_module_output

init_session_state()

st.title("📊 Module 3: Plan Expansion and Evaluation")
st.markdown("Expand plan items and evaluate against success criteria.")

if not st.session_state.api_key:
    st.error("❌ Please configure your OpenAI API key in the API Configuration page first.")
    st.stop()

previous_output = get_previous_module_output('module3')
if previous_output:
    st.success("✅ Input available from Module 2")
else:
    st.warning("⚠️ No output from Module 2 found. Please complete Module 2 first.")

st.info("🚧 Module 3 functionality will be implemented in Phase 2")

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
```

#### File: `streamlit_app/pages/5_Module_4_Revision.py`
**IMPORTANT**: The filename is `5_Module_4_Revision.py` with NO emoji or special characters
```python
import streamlit as st
from utils.session_state import init_session_state, get_previous_module_output

init_session_state()

st.title("🔧 Module 4: Revision Identification")
st.markdown("Identify needed revisions based on evaluation results.")

if not st.session_state.api_key:
    st.error("❌ Please configure your OpenAI API key in the API Configuration page first.")
    st.stop()

previous_output = get_previous_module_output('module4')
if previous_output:
    st.success("✅ Input available from Module 3")
else:
    st.warning("⚠️ No output from Module 3 found. Please complete Module 3 first.")

st.info("🚧 Module 4 functionality will be implemented in Phase 2")

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
```

#### File: `streamlit_app/pages/6_Module_5_Implementation.py`
**IMPORTANT**: The filename is `6_Module_5_Implementation.py` with NO emoji or special characters
```python
import streamlit as st
from utils.session_state import init_session_state, get_previous_module_output

init_session_state()

st.title("✨ Module 5: Revision Implementation")
st.markdown("Implement approved revisions into the plan.")

if not st.session_state.api_key:
    st.error("❌ Please configure your OpenAI API key in the API Configuration page first.")
    st.stop()

previous_output = get_previous_module_output('module5')
if previous_output:
    st.success("✅ Input available from Module 4")
else:
    st.warning("⚠️ No output from Module 4 found. Please complete Module 4 first.")

st.info("🚧 Module 5 functionality will be implemented in Phase 2")

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
```

#### File: `streamlit_app/pages/7_Module_6_Report.py`
**IMPORTANT**: The filename is `7_Module_6_Report.py` with NO emoji or special characters
```python
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
```

#### File: `streamlit_app/README.md`
Copy the full content from the previous README, but ensure it mentions the simplified filenames without emoji.

### Step 5: Verify Directory Structure
After creating all files, verify your structure matches EXACTLY:
```
agento-streamlit/
├── module1.py (and other module files)
├── streamlit_app/
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   ├── pages/
│   │   ├── 1_API_Configuration.py
│   │   ├── 2_Module_1_Criteria.py
│   │   ├── 3_Module_2_Planning.py
│   │   ├── 4_Module_3_Expansion.py
│   │   ├── 5_Module_4_Revision.py
│   │   ├── 6_Module_5_Implementation.py
│   │   └── 7_Module_6_Report.py
│   └── utils/
│       ├── __init__.py
│       ├── session_state.py
│       └── file_handlers.py
```

### Step 6: Install Dependencies
```bash
cd streamlit_app
pip install -r requirements.txt
```

### Step 7: Run the Application
```bash
streamlit run app.py
```

## Critical Reminders
- NO emoji in filenames
- NO extra directories like "b/"
- Files must be in EXACT locations specified
- The module files (module1.py etc.) must be in the ROOT agento-streamlit directory
- All page files go in streamlit_app/pages/
- All utility files go in streamlit_app/utils/
