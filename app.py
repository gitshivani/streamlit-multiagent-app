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
def verify_api_key(provider, api_key):
    try:
        if provider == "openai":
            client = openai.OpenAI(api_key=api_key)
            client.models.list()
            return True, "OpenAI API key verified successfully"
        elif provider == "anthropic":
            client = Anthropic(api_key=api_key)
            return True, "Anthropic API key format accepted"
        elif provider == "gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            return True, "Google Gemini API key format accepted"
        elif provider in ["cohere", "mistral", "llama", "deepseek"]:
            if len(api_key) >= 20:
                return True, f"{provider.capitalize()} API key format accepted"
            else:
                return False, f"{provider.capitalize()} API key seems too short"
        else:
            return False, "Unknown provider"
    except Exception as e:
        return False, f"API verification failed: {str(e)}"

def generate_with_llm(provider, api_key, prompt, model=None):
    try:
        if provider == "openai":
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model or "gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            return True, response.choices[0].message.content
        elif provider == "anthropic":
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model or "claude-3-haiku-20240307",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            try:
                return True, response.content[0].text
            except (IndexError, AttributeError):
                return True, str(response)
        elif provider == "gemini":
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(model or "gemini-1.5-flash")
            response = gemini_model.generate_content(prompt)
            return True, response.text
        elif provider == "cohere":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model or "command",
                "prompt": prompt,
                "max_tokens": 1024
            }
            response = requests.post("https://api.cohere.ai/v1/generate", headers=headers, json=payload)
            if response.status_code == 200:
                return True, response.json().get("generations", [{}])[0].get("text", "")
            else:
                return False, f"Cohere API error: {response.text}"
        elif provider == "mistral":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model or "mistral-small-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024
            }
            response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                return True, response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                return False, f"Mistral API error: {response.text}"
        elif provider in ["llama", "deepseek"]:
            return False, f"The {provider} API integration is not available in this demo version. Please use one of the other providers."
        else:
            return False, "Unknown provider"
    except Exception as e:
        return False, f"Generation failed: {str(e)}"

# ✅ FIX: Define llm_providers BEFORE calling main()
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

# Main app function
def main():
    # (... your unchanged main() implementation ...)
    pass  # ← Replace with your main() logic

if __name__ == "__main__":
    main()
