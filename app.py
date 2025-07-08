import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
import httpx
import google.generativeai as genai
from anthropic import Anthropic
import openai
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API URL for your FastAPI backend
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Shivani Gupta's Multi-LLM Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stTextInput, .stSelectbox, .stTextarea {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #e0e4e8;
        padding: 10px;
    }
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    .output-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    h1, h2, h3 {
        color: #4F46E5;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .error-message {
        color: #D32F2F;
        background-color: #FFEBEE;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #D32F2F;
    }
    .success-message {
        color: #2E7D32;
        background-color: #E8F5E9;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #2E7D32;
    }
    .warning-message {
        color: #FF6D00;
        background-color: #FFF3E0;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #FF6D00;
    }
    .gradient-text {
        font-weight: 800;
        color: #4F46E5;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'api_verified' not in st.session_state:
    st.session_state.api_verified = {}

if 'active_provider' not in st.session_state:
    st.session_state.active_provider = None

# Helper functions
# (... No change in verify_api_key and generate_with_llm ...)
# Provider configurations
# (... No change in llm_providers ...)

# Main app function
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 class='gradient-text'>🚀 Shivani Gupta's Multi-LLM Hub</h2>", unsafe_allow_html=True)

        st.markdown("### 🤖 Select LLM Provider")

        # Provider selection
        for provider_id, provider_info in llm_providers.items():
            provider_selected = st.session_state.active_provider == provider_id

            if st.button(
                f"{provider_info['name']}",
                key=f"provider_{provider_id}",
                help=provider_info['description'],
                use_container_width=True,
                type="primary" if provider_selected else "secondary"
            ):
                st.session_state.active_provider = provider_id
                st.rerun()

            # Show verification status if available
            if provider_id in st.session_state.api_verified:
                if st.session_state.api_verified[provider_id]:
                    st.success(f"{provider_info['name']} verified ✅")
                else:
                    st.error(f"{provider_info['name']} invalid ❌")

        st.markdown("---")

        # If a provider is selected, show API key input and model selection
        if st.session_state.active_provider:
            provider = st.session_state.active_provider
            provider_info = llm_providers[provider]

            st.markdown(f"### {provider_info['name']} Configuration")

            # API Key input
            api_key = st.text_input(
                f"{provider_info['name']} API Key",
                type="password",
                key=f"api_key_{provider}"
            )

            # Verify button
            if api_key:
                if st.button("Verify API Key", key=f"verify_{provider}"):
                    with st.spinner("Verifying API key..."):
                        is_valid, message = verify_api_key(provider, api_key)
                        if is_valid:
                            st.session_state.api_verified[provider] = True
                            st.success(message)
                        else:
                            st.session_state.api_verified[provider] = False
                            st.error(message)

            # Model selection if provider is verified
            if provider in st.session_state.api_verified and st.session_state.api_verified[provider]:
                st.markdown("### Model Selection")

                selected_model = st.selectbox(
                    "Choose Model",
                    options=list(provider_info["models"].keys()),
                    format_func=lambda x: provider_info["models"][x],
                    key=f"model_{provider}"
                )

                # Display model info
                st.info(provider_info["models"][selected_model])

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This application was created by **Shivani Gupta** to explore the capabilities of various LLM providers
        for story generation, text summarization, and translation tasks.
        """)
        st.markdown("Built with Streamlit & Python 💻")

    # Main content
    st.markdown("<h1 class='gradient-text'>Welcome to Shivani’s AI Hub 🚀</h1>", unsafe_allow_html=True)
    st.markdown("Generate creative stories, summarize long text, and translate between languages using various AI models!")

    # (... rest of your unchanged code for tabs and functions ...)
    # Place entire Story Generator / Summarizer / Translator tab code here — unchanged

if __name__ == "__main__":
    main()