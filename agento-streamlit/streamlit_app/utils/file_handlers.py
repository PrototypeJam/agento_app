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
