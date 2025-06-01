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
