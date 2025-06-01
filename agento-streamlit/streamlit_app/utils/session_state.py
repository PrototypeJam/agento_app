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
