import os
import streamlit as st
import google.generativeai as genai
import json
from datetime import datetime
from src.utils import load_commit_data, format_commit_data

# Set page configuration
st.set_page_config(page_title="Git Repo AI Assistant", layout="wide")

# --- CONFIGURATION ---
# Configure the Gemini API key
API_KEY = st.secrets.get("GEMINI_API_KEY", None)
if not API_KEY:
    st.error("Gemini API key not found. Please set GEMINI_API_KEY in Streamlit secrets.")
    st.stop()
genai.configure(api_key=API_KEY)

GEMINI_MODEL = "gemini-1.5-flash"
model = genai.GenerativeModel(GEMINI_MODEL)

# --- STREAMLIT UI SETUP ---
st.title("📘 Git Repository Chatbot")
st.markdown("Ask about what happened in your repo: what, who, and when.")

# Sidebar
st.sidebar.title("💬 AI Git Assistant")

# Home page chat functionality
repo_name = st.sidebar.text_input("Repository name", value="ChefTreffHackFIChallenge")
question = st.text_input("Ask a question", placeholder="e.g., What changed recently?")
ask_button = st.sidebar.button("Ask")
json_path = f"data/{repo_name}_commit_DB.json"

# Load commit data (cached)
try:
    commit_data = load_commit_data(json_path)
except FileNotFoundError:
    st.error(f"File not found: {json_path}")
    st.write(f"Current working directory: {os.getcwd()}")
    st.stop()

# Initialize Gemini chat model
chat = model.start_chat()
with open('prompts/read_git_summaries_init_llm', 'r') as f:
    init_prompt = f.read()

init_response = chat.send_message(init_prompt)

# Handle user question
if ask_button and question:
    with st.spinner("Thinking..."):
        prompt = (
            f"{format_commit_data(commit_data)}\n\n"
            f"User question: {question}\n"
            f"Only answer based on the commit summaries and metadata. Use commonly known language understandable by non-technical stakeholders"
        )
        response = chat.send_message(prompt)
    st.success("AI Response")
    st.write(response.text)
else:
    st.info("Ask me anything about this Git repository")