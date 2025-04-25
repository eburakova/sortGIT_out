import os
import streamlit as st
import google.generativeai as genai
from src.utils import load_commit_data, format_commit_data, format_git_time

from src.sanity_checks import define_processing_mode
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
#question = st.text_input("Ask a question", placeholder="e.g., What changed recently?")
#ask_button = st.sidebar.button("Ask")
json_path = f"data/{repo_name}_commit_DB.json"

# Load commit data (cached)
try:
    commit_data = load_commit_data(json_path)
except FileNotFoundError:
    st.error(f"File not found: {json_path}")
    st.write(f"Current working directory: {os.getcwd()}")
    st.stop()

data_processing_mode = define_processing_mode(commit_data)
if data_processing_mode == 'Summary':
    st.write("Running AI in data-saving mode")
if data_processing_mode == 'Full':
    st.write("Git history is within limits, running AI in with full data.")
if data_processing_mode == 'Prefilter':
    st.write("Running AI in extreme data-saving mode (this may impact accuracy! Prefilter the data manually if possible)")

    col1, col2, col3 = st.columns(3)
    with col1:
        search_term = st.text_input("Search commits", placeholder="Search by message, author, or files")
    with col2:
        start_date = st.date_input("Start date", value=None)
    with col3:
        end_date = st.date_input("End date", value=None)
    if start_date and end_date:
        if start_date <= end_date:
            st.success(f"Selected date range: {start_date} to {end_date}")
            try:
                filtered_commits_by_date = [commit for commit in commit_data if
                                        format_git_time(commit['date']) >= start_date and format_git_time(commit['date']) <= end_date]
                commit_data = filtered_commits_by_date
            except TypeError as e:
                st.error(f"Error: malformatted dates, can't filter {commit_data[0]['date']}, {e}")
        else:
            st.error("Error: End date must be after start date")

#######
# Initialize Gemini chat model
chat = model.start_chat()
with open('prompts/read_git_summaries_init_llm', 'r') as f:
    init_prompt = f.read()

init_response = chat.send_message(init_prompt)

with st.form(key="question_form", clear_on_submit=True):
    question = st.text_input("Ask a question", placeholder="e.g., What changed recently?")
    submit_button = st.form_submit_button("Ask", use_container_width=True)

    # You can hide the submit button if you want to rely solely on Enter key
    # To hide it, use CSS:
    st.markdown("""<style>.stButton button {display: none;}</style>""", unsafe_allow_html=True) # Consider to fix

# Then replace your if statement with this:
if submit_button and question:
    with st.spinner("Thinking..."):
        # Your existing code for processing the question
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