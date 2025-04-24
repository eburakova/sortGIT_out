import streamlit as st
import google.generativeai as genai
from datetime import datetime, date, timedelta, timezone # Added timezone
from git import Repo
import json


# ---- CONFIG ----
GEMINI_MODEL = "gemini-1.5-flash"
GITHUB_API_BASE = "https://api.github.com/repos"
MAX_HISTORY_TURNS = 5 # Max number of user/assistant turn pairs to include in history context
COMMITS_PER_PAGE = 30 # Changed as requested

# Define supported languages for the UI and translation prompt
SUPPORTED_LANGUAGES = {
    "English": "English", "German": "German", "French": "French",
    "Spanish": "Spanish", "Japanese": "Japanese", "Korean": "Korean",
    "Mandarin Chinese": "Mandarin Chinese"
}

model = genai.GenerativeModel(GEMINI_MODEL)

@st.cache_data(ttl=86400)
def get_repo_log(repo_path) -> dict:
    repo = Repo('../commit-messages-guide')  # or path to your repo
    commits_data = []

    for commit in repo.iter_commits():
        commit_info = {
            "commit": commit.hexsha,
            "author": str(commit.author.name),
            "author_email": str(commit.author.email),
            "date": str(commit.committed_datetime),
            "message": str(commit.message),
            "files_changed": []
        }

        for diff in commit.diff(None, create_patch=True):
            if diff.a_path and diff.diff:
                diff_raw = diff.diff.decode(errors="ignore", encoding="utf-8")

                commit_info["files_changed"].append({
                    "file": diff.a_path,
                    "diff": diff_raw,
                   # "diff_processed": extract_diff_lines(diff_raw)
                })

        commits_data.append(commit_info)

    # Save or print as JSON
    with open('commits_DB.json', 'w') as f:
        json.dump(commits_data, f, indent=2)

@st.cache_data(ttl=86400)
def get_commit_details(commit_hash, repo_log):
    pass


# ---- Helper Functions for AI Interaction ---- #
@st.cache_data(ttl=86400)
def summarize_commit(_commit_data, target_language="English"):
    """Summarizes a commit using the Gemini model and translates if needed."""
    commit_info = _commit_data.get('commit', {})
    author_info = commit_info.get('author', {})
    commit_message = commit_info.get('message', 'N/A')
    author = author_info.get('name', 'N/A')
    date_str = author_info.get('date', 'N/A')
    sha = _commit_data.get('sha', 'N/A')
    prompt = f"""
You are an expert AI assistant specializing in code analysis.
Analyze and summarize the following Git commit in **simple, non-technical language**.
Focus on the **what** and **why** of the change, understandable by someone with minimal programming background. Keep the summary concise (2-3 sentences).

Commit SHA: {sha}
Commit Message: {commit_message}
Author: {author}
Date: {date_str}
"""
    if target_language != "English": prompt += f"\n\nPlease provide the summary exclusively in {target_language}."
    else: prompt += f"\n\nPlease provide the summary in English."
    try:
        response = model.generate_content(prompt)
        if response.parts: return response.text
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             reason = response.prompt_feedback.block_reason.name.replace("_", " ").lower()
             return f"Could not generate summary for {sha[:7]}. Blocked due to {reason} content."
        else: return f"Could not generate summary for {sha[:7]}. Empty/invalid response from AI."
    except Exception as e: return f"Error generating summary for {sha[:7]}: {e}"

def format_commit_context(commits_on_page):
    """Formats commit data *from the current page* into context."""
    if not commits_on_page: return "No commit data available for the current page."
    MAX_COMMITS_FOR_CONTEXT = COMMITS_PER_PAGE
    context_str = f"Commit data for the current page (up to {MAX_COMMITS_FOR_CONTEXT} commits shown):\n\n"
    commits_to_include = commits_on_page[:MAX_COMMITS_FOR_CONTEXT]
    for commit in commits_to_include:
        sha = commit.get('sha', 'N/A')
        commit_info = commit.get('commit', {})
        author_info = commit_info.get('author', {})
        author = author_info.get('name', 'N/A')
        date_str = author_info.get('date', 'N/A')
        message = commit_info.get('message', 'N/A')
        context_str += f"Commit SHA: {sha}\nAuthor: {author}\nDate: {date_str}\nMessage:\n{message}\n---\n"
    return context_str

def format_chat_history(chat_history):
    """Formats the chat history list into a string for the AI prompt, truncating if necessary."""
    if not chat_history: return "No previous conversation history in this session."
    turns_to_include = min(len(chat_history), MAX_HISTORY_TURNS * 2)
    relevant_history = chat_history[-turns_to_include:]
    formatted_history = ""
    for msg in relevant_history:
        role = "User" if msg.get('role') == 'user' else "Assistant"
        formatted_history += f"{role}: {msg.get('content', '')}\n"
    if len(chat_history) > turns_to_include: formatted_history = "[...truncated history...]\n" + formatted_history
    return formatted_history.strip()

def answer_question(question, commit_context, chat_history_list):
    """Uses Gemini to answer a question based on commit context and chat history."""
    if not question: return "Please ask a question."
    if not commit_context or commit_context == "No commit data available for the current page.": return "Cannot answer: No commit data."
    formatted_chat_history = format_chat_history(chat_history_list)
    prompt = f"""
You are a helpful AI assistant acting as a communicator who translates technical Git commit information for a **non-technical audience**.
Your goal is to answer the user's **latest question** based *only* on the **Provided Commit Data** below (which represents a specific page/section of commits), considering the **Ongoing Conversation History** for context. Explain the answer in **simple, everyday language**.

**Instructions:**
* Analyze the **Provided Commit Data**. This is your *primary source*.
* Review the **Ongoing Conversation History** for context.
* Answer the **User's Latest Question** accurately based *only* on the commit data.
* **Explain simply and concisely.** Avoid jargon. Focus on 'what' and 'why'.
* Use history to resolve ambiguities, but **do not base factual answers on history.** Base facts *only* on the Provided Commit Data.
* If technical terms are unavoidable, explain them simply.
* If the answer cannot be found in the commit data, state that clearly.

**Provided Commit Data (Current Page):**
{commit_context}
---
**Ongoing Conversation History:**
{formatted_chat_history}
---
**User's Latest Question:**
{question}
---
**Answer:**
"""
    try:
        response = model.generate_content(prompt)
        if response.parts: return response.text
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             reason = response.prompt_feedback.block_reason.name.replace("_", " ").lower()
             return f"Sorry, couldn't generate answer. Blocked due to {reason} content."
        else: return "Sorry, received an empty response from the AI."
    except Exception as e: return f"Sorry, an error occurred generating the answer: {e}"

# ---- Session State Initialization ----
if 'process_request' not in st.session_state: st.session_state.process_request = False
if 'current_page_number' not in st.session_state: st.session_state.current_page_number = 1
if 'total_api_pages' not in st.session_state: st.session_state.total_api_pages = 1
if 'active_sha_filter' not in st.session_state: st.session_state.active_sha_filter = None
if 'active_start_date' not in st.session_state: st.session_state.active_start_date = None
if 'active_end_date' not in st.session_state: st.session_state.active_end_date = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'repo_url' not in st.session_state: st.session_state.repo_path = ""
if 'owner' not in st.session_state: st.session_state.owner = None
if 'repo' not in st.session_state: st.session_state.repo = None
