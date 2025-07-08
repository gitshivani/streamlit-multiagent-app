# === [All your original imports and setup code] ===
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

# Streamlit page setup
st.set_page_config(
    page_title="Shivani Gupta's Multi-LLM Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS styling
st.markdown("""<style>/* ... your full CSS block unchanged ... */</style>""", unsafe_allow_html=True)

# Streamlit session state
if 'api_verified' not in st.session_state:
    st.session_state.api_verified = {}

if 'active_provider' not in st.session_state:
    st.session_state.active_provider = None

# === [llm_providers MUST be defined before main() is called] ===
llm_providers = {
    "openai": {
        "name": "OpenAI",
        "description": "GPT models from OpenAI (GPT-3.5, GPT-4)",
        "models": {
            "gpt-3.5-turbo": "GPT-3.5 Turbo (Fast, Affordable)",
            "gpt-4o": "GPT-4o (Powerful, Multi-modal)",
            "gpt-4o-mini": "GPT-4o Mini (Balanced)"
        }
    },
    "anthropic": {
        "name": "Anthropic",
        "description": "Claude models from Anthropic",
        "models": {
            "claude-3-haiku-20240307": "Claude 3 Haiku (Fast)",
            "claude-3-sonnet-20240229": "Claude 3 Sonnet (Balanced)",
            "claude-3-opus-20240229": "Claude 3 Opus (Powerful)"
        }
    },
    "gemini": {
        "name": "Google",
        "description": "Gemini models from Google",
        "models": {
            "gemini-1.5-flash": "Gemini 1.5 Flash (Fast)",
            "gemini-1.5-pro": "Gemini 1.5 Pro (Balanced)"
        }
    },
    "mistral": {
        "name": "Mistral AI",
        "description": "Models from Mistral AI",
        "models": {
            "mistral-small-latest": "Mistral Small (Fast)",
            "mistral-medium-latest": "Mistral Medium (Balanced)",
            "mistral-large-latest": "Mistral Large (Powerful)"
        }
    },
    "cohere": {
        "name": "Cohere",
        "description": "Models from Cohere",
        "models": {
            "command": "Command (General)",
            "command-light": "Command Light (Fast)",
            "command-r": "Command-R (Robust)"
        }
    }
}

# === [Your verify and generate helper functions — unchanged] ===
# (you already pasted them fully earlier — they go here untouched)

# === ✅ Your full original main() function starts here ===
def main():
    with st.sidebar:
        st.markdown("<h2 class='gradient-text'>🚀 Shivani Gupta's Multi-LLM Hub</h2>", unsafe_allow_html=True)
        st.markdown("### 🤖 Select LLM Provider")

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

            if provider_id in st.session_state.api_verified:
                if st.session_state.api_verified[provider_id]:
                    st.success(f"{provider_info['name']} verified ✅")
                else:
                    st.error(f"{provider_info['name']} invalid ❌")

        st.markdown("---")

        if st.session_state.active_provider:
            provider = st.session_state.active_provider
            provider_info = llm_providers[provider]

            st.markdown(f"### {provider_info['name']} Configuration")

            api_key = st.text_input(
                f"{provider_info['name']} API Key",
                type="password",
                key=f"api_key_{provider}"
            )

            if api_key:
                if st.button("Verify API Key", key=f"verify_{provider}"):
                    with st.spinner("Verifying API key..."):
                        is_valid, message = verify_api_key(provider, api_key)
                        st.session_state.api_verified[provider] = is_valid
                        if is_valid:
                            st.success(message)
                        else:
                            st.error(message)

            if provider in st.session_state.api_verified and st.session_state.api_verified[provider]:
                st.markdown("### Model Selection")

                selected_model = st.selectbox(
                    "Choose Model",
                    options=list(provider_info["models"].keys()),
                    format_func=lambda x: provider_info["models"][x],
                    key=f"model_{provider}"
                )

                st.info(provider_info["models"][selected_model])

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This application was created by **Shivani Gupta** to explore the capabilities of various LLM providers
        for story generation, text summarization, and translation tasks.
        """)
        st.markdown("Built with Streamlit & Python 💻")

    st.markdown("<h1 class='gradient-text'>Welcome to Shivani’s AI Hub 🚀</h1>", unsafe_allow_html=True)
    st.markdown("Generate creative stories, summarize long text, and translate between languages using various AI models!")

    if not st.session_state.active_provider:
        st.warning("Please select an LLM provider from the sidebar to get started.")
        return

    provider = st.session_state.active_provider
    provider_info = llm_providers[provider]

    if provider not in st.session_state.api_verified or not st.session_state.api_verified[provider]:
        st.warning(f"Please enter and verify your {provider_info['name']} API key in the sidebar.")
        return

    api_key = st.session_state[f"api_key_{provider}"]
    model = st.session_state.get(f"model_{provider}", list(provider_info["models"].keys())[0])

    tabs = st.tabs(["📝 Story Generator", "📚 Text Summarizer", "🌐 Translator"])

    with tabs[0]:
        st.markdown("### 📝 Generate Creative Stories")
        st.markdown("Enter a title or topic, and let the AI craft a unique story for you!")

        col1, col2 = st.columns([3, 1])
        with col1:
            story_title = st.text_input("Story Title or Topic", placeholder="Enter a title or topic for your story")
        with col2:
            generate_button = st.button("Generate Story 🚀", use_container_width=True)

        if story_title and generate_button:
            if len(story_title) < 3:
                st.error("Title is too short. Please provide at least 3 characters.")
            else:
                with st.spinner("Generating your story..."):
                    prompt = f"Generate a creative, engaging story about the following topic or title: '{story_title}'. Make it approximately 500 words long with a clear beginning, middle, and end."
                    success, result = generate_with_llm(provider, api_key, prompt, model)
                    if success:
                        st.markdown(f"### {story_title}")
                        st.write(result)
                    else:
                        st.error(result)

    with tabs[1]:
        st.markdown("### 📚 Summarize Long Text")
        st.markdown("Paste your long text below, and get a concise summary!")

        text_to_summarize = st.text_area("Text to Summarize", height=200, placeholder="Paste your long text here (minimum 100 characters)")
        summarize_button = st.button("Summarize Text 📝", use_container_width=True)

        if text_to_summarize and summarize_button:
            if len(text_to_summarize) < 100:
                st.error("Text is too short. Please provide at least 100 characters for a meaningful summary.")
            else:
                with st.spinner("Summarizing your text..."):
                    prompt = f"Summarize the following text concisely, capturing the main points and important details:\n\n{text_to_summarize}"
                    success, result = generate_with_llm(provider, api_key, prompt, model)
                    if success:
                        st.markdown("### Summary")
                        st.write(result)
                    else:
                        st.error(result)

    with tabs[2]:
        st.markdown("### 🌐 Translate Text")
        st.markdown("Enter text and select a target language for translation!")

        text_to_translate = st.text_area("Text to Translate", height=150, placeholder="Enter text to translate")

        col1, col2 = st.columns([3, 1])
        with col1:
            languages = [
                "Arabic", "Bengali", "Chinese (Simplified)", "Chinese (Traditional)",
                "Dutch", "English", "French", "German", "Greek", "Hindi", "Indonesian",
                "Italian", "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
                "Swahili", "Tamil", "Thai", "Turkish", "Ukrainian", "Vietnamese"
            ]
            target_language = st.selectbox("Target Language", languages)

        with col2:
            translate_button = st.button("Translate 🌍", use_container_width=True)

        if text_to_translate and translate_button:
            with st.spinner(f"Translating to {target_language}..."):
                prompt = f"Translate the following text to {target_language}. Maintain the original meaning and tone as closely as possible:\n\n{text_to_translate}"
                success, result = generate_with_llm(provider, api_key, prompt, model)
                if success:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Original Text")
                        st.write(text_to_translate)
                    with col2:
                        st.markdown(f"### Translated Text ({target_language})")
                        st.write(result)
                else:
                    st.error(result)

# === Run the app ===
if __name__ == "__main__":
    main()
