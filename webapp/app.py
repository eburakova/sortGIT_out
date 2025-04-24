import streamlit as st
import requests
import os
import google.generativeai as genai
from datetime import datetime, date, timedelta, timezone # Added timezone
import re
import math

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

# ---- AUTH ----
# Ensure you have GEMINI_API_KEY and GITHUB_TOKEN defined in /.streamlit/secrets.toml
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY missing.")
    if not GITHUB_TOKEN: raise ValueError("GITHUB_TOKEN missing.")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
except KeyError as e:
    st.error(f"Configuration Error: Secret key '{e}' not found in secrets.")
    st.stop()
except ValueError as e:
    st.error(f"Configuration Error: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during setup: {e}")
    st.stop()

# ---- HELPER FUNCTIONS ----
def parse_repo_url(url):
    """Parses GitHub URL to extract owner and repo."""
    pattern = r"https://github\.com/([^/]+)/([^/]+?)(\.git)?/?$"
    match = re.search(pattern, url)
    if match: return match.group(1), match.group(2)
    return None, None

def parse_link_header(headers):
    """Parses the Link header from GitHub API response to find the last page number."""
    link_header = headers.get('Link')
    if not link_header: return None
    links = {}
    parts = link_header.split(',')
    for part in parts:
        section = part.split(';')
        if len(section) != 2: continue
        try:
            url = section[0].replace('<', '').replace('>', '').strip()
            rel = section[1].strip().replace('rel="', '').replace('"', '')
            links[rel] = url
        except: continue # Ignore malformed parts
    if 'last' in links:
        try:
            match = re.search(r'[?&]page=(\d+)', links['last'])
            if match: return int(match.group(1))
        except: return None
    return None # Return None if 'last' link is definitively not found

def parse_commit_date(date_string):
    """Parses GitHub's ISO 8601 date string into a date object."""
    if not date_string or not isinstance(date_string, str): return None
    try:
        if date_string.endswith('Z'): date_string = date_string[:-1] + '+00:00'
        return datetime.fromisoformat(date_string).date()
    except (ValueError, TypeError): return None

# ---- API FUNCTIONS ----
@st.cache_data(ttl=600) # Cache API page results for 10 minutes
def get_commits_page(owner, repo, github_token, page_num, per_page, sha_filter=None, start_date_iso=None, end_date_iso=None):
    """Fetches a specific page of commits from the repository using API filters."""
    if not owner or not repo:
        return None, "Invalid repository owner or name.", None

    api_url = f"{GITHUB_API_BASE}/{owner}/{repo}/commits"
    headers = {"Authorization": f"Bearer {github_token}", "Accept": "application/vnd.github.v3+json"}
    params = {'page': page_num, 'per_page': per_page}

    if sha_filter: params['sha'] = sha_filter
    if start_date_iso: params['since'] = start_date_iso
    if end_date_iso: params['until'] = end_date_iso

    try:
        res = requests.get(api_url, headers=headers, params=params, timeout=20)
        res.raise_for_status()
        commits_on_page = res.json()
        total_pages = parse_link_header(res.headers)

        if total_pages is None:
            if commits_on_page and len(commits_on_page) < per_page: total_pages = page_num
            elif not commits_on_page and page_num > 1: total_pages = page_num - 1
            elif not commits_on_page and page_num == 1: total_pages = 1
            elif page_num > 1 and len(commits_on_page) == per_page : total_pages = page_num # Fallback

        if not isinstance(commits_on_page, list):
             return None, f"Unexpected API response format: {type(commits_on_page)}", None
        return commits_on_page, None, total_pages

    except requests.exceptions.RequestException as e:
        error_message = f"API error fetching page {page_num}: {e}"
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            if status_code == 404: error_message = f"Repo '{owner}/{repo}' not found or filters invalid (404)."
            elif status_code == 401: error_message = f"Auth failed (401). Check token."
            elif status_code == 403: error_message = f"Access forbidden (403). Check permissions/rate limits."
            elif status_code == 422: error_message = f"Unprocessable Entity (422). Invalid filter combination?"
            else: error_message += f" (Status code: {status_code})"
        return None, error_message, None
    except Exception as e:
        return None, f"Unexpected error fetching page {page_num}: {e}", None


@st.cache_data(ttl=86400)
def get_commit_details(owner, repo, github_token, commit_sha):
    """Fetches detailed information for a single commit using authentication."""
    if not owner or not repo or not commit_sha: return None, "Missing owner, repo, or SHA."
    api_url = f"{GITHUB_API_BASE}/{owner}/{repo}/commits/{commit_sha}"
    headers = {"Authorization": f"Bearer {github_token}", "Accept": "application/vnd.github.v3+json"}
    try:
        res = requests.get(api_url, headers=headers, timeout=15)
        res.raise_for_status()
        commit_data = res.json()
        if 'files' in commit_data: return commit_data, None
        else: return commit_data, "Commit data fetched, but no file change details included."
    except requests.exceptions.RequestException as e:
        error_message = f"API error fetching details for {commit_sha[:7]}: {e}"
        if hasattr(e, 'response') and e.response is not None:
             status_code = e.response.status_code
             if status_code == 404: error_message = f"Commit {commit_sha} not found (404)."
             elif status_code == 401: error_message = f"Auth failed (401)."
             elif status_code == 403: error_message = f"Access forbidden for commit details (403)."
             elif status_code == 422: error_message = f"Cannot retrieve diff for commit {commit_sha[:7]} (422)."
             else: error_message += f" (Status code: {status_code})"
        return None, error_message
    except Exception as e:
        return None, f"Unexpected error fetching details for {commit_sha[:7]}: {e}"

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
if 'repo_url' not in st.session_state: st.session_state.repo_url = ""
if 'owner' not in st.session_state: st.session_state.owner = None
if 'repo' not in st.session_state: st.session_state.repo = None


# ---- UI ----
st.set_page_config(layout="wide", page_title="AI Commit Summarizer & Q&A")
st.title("🚀 AI-Powered GitHub Commit Summarizer & Q&A")
st.markdown("Get summaries and ask questions about commits from a GitHub repository (fetches commits per page).")

# --- Input Area ---
col1, col2 = st.columns([3, 1])
with col1:
    st.session_state.repo_url = st.text_input(
        "Paste a Public or Private GitHub Repository URL:",
        placeholder="e.g., https://github.com/streamlit/streamlit",
        key="repo_url_input", value=st.session_state.repo_url,
        help="Enter the full URL of the GitHub repository."
    )
    repo_url = st.session_state.repo_url
    valid_url = False
    if repo_url:
        owner_in, repo_in = parse_repo_url(repo_url)
        if owner_in and repo_in:
            if owner_in != st.session_state.owner or repo_in != st.session_state.repo:
                 st.session_state.owner = owner_in
                 st.session_state.repo = repo_in
                 st.session_state.process_request = False
                 st.session_state.chat_history = []
                 st.session_state.current_page_number = 1
                 st.session_state.total_api_pages = 1
                 st.session_state.active_sha_filter = None
                 st.session_state.active_start_date = None
                 st.session_state.active_end_date = None
                 st.cache_data.clear()
            valid_url = True
        else:
            if st.session_state.owner or st.session_state.repo:
                 st.session_state.owner = None; st.session_state.repo = None
                 st.session_state.process_request = False; st.session_state.chat_history = []
                 st.session_state.current_page_number = 1; st.session_state.total_api_pages = 1
            if repo_url: st.warning("Invalid GitHub URL format.", icon="⚠️")

with col2:
    selected_language = st.selectbox(
        "Select Summary Language:", options=list(SUPPORTED_LANGUAGES.keys()), index=0, key="language_select",
        help="Choose the language for the AI-generated commit summaries."
    )

# --- Filtering Area ---
st.subheader("🔎 Filter Commits (Optional)")
filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    sha_input = st.text_input("Filter by Branch/Tag/SHA:", placeholder="e.g., main or v1.0 or a1b2c3d", key="sha_filter", help="Enter Branch, Tag, or Commit SHA.").strip()
with filter_col2:
    start_date_input = st.date_input("Filter From Date:", value=None, key="start_date_filter", help="Show commits from this date (inclusive).")
with filter_col3:
    today_date = datetime.now(timezone.utc).date()
    end_date_input = st.date_input("Filter To Date:", value=None, key="end_date_filter", min_value=start_date_input if start_date_input else None, help="Show commits up to this date (inclusive).")

is_date_range_invalid = bool(start_date_input and end_date_input and start_date_input > end_date_input)
if is_date_range_invalid: st.warning("Warning: Start date is after end date.", icon="⚠️")

# --- Process Button ---
button_disabled_state = not valid_url or is_date_range_invalid
process_button = st.button(
    "Get Commits & Apply Filters", key="process_button", type="primary", disabled=button_disabled_state
)

# --- Logic after Get Commits Button Click ---
if process_button:
    if st.session_state.owner and st.session_state.repo:
        st.session_state.active_sha_filter = sha_input if sha_input else None
        st.session_state.active_start_date = start_date_input
        st.session_state.active_end_date = end_date_input
        st.session_state.process_request = True
        st.session_state.current_page_number = 1
        st.session_state.total_api_pages = 1
        st.session_state.chat_history = []
        st.info("Filters applied. Commit data will be fetched for the selected page.")
        st.rerun()


# --- Display Section (Pagination, Summaries, Q&A) ---
if st.session_state.process_request and st.session_state.owner and st.session_state.repo:

    st.markdown("---")
    st.subheader("Browse Commits")

    page_to_fetch = st.session_state.current_page_number
    owner = st.session_state.owner
    repo = st.session_state.repo
    sha_filter = st.session_state.active_sha_filter
    start_date = st.session_state.active_start_date
    end_date = st.session_state.active_end_date

    start_date_iso = None
    if start_date: start_date_iso = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0, tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z')
    end_date_iso = None
    if end_date: end_date_iso = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z')

    commits_on_current_page = None
    error = None
    total_api_pages_from_fetch = None

    with st.spinner(f"Fetching commits for page {page_to_fetch}..."):
        commits_on_current_page, error, total_api_pages_from_fetch = get_commits_page(
            owner, repo, GITHUB_TOKEN, page_to_fetch, COMMITS_PER_PAGE,
            sha_filter, start_date_iso, end_date_iso
        )

    if error is None and total_api_pages_from_fetch is not None: st.session_state.total_api_pages = total_api_pages_from_fetch
    elif error is None and total_api_pages_from_fetch is None and not commits_on_current_page and page_to_fetch == 1: st.session_state.total_api_pages = 1
    total_pages_display = st.session_state.total_api_pages

    # --- Pagination Controls ---
    if total_pages_display > 1 or page_to_fetch > 1 :
        prev_page_number = st.session_state.current_page_number
        pcol1, pcol2 = st.columns([1, 3])
        with pcol1:
            current_page_clamped = max(1, min(st.session_state.current_page_number, total_pages_display if total_pages_display else st.session_state.current_page_number))
            if current_page_clamped != st.session_state.current_page_number: st.session_state.current_page_number = current_page_clamped

            new_page_num = st.number_input(
                 f"Page (1-{total_pages_display if total_pages_display else '?'})",
                 min_value=1,
                 max_value=total_pages_display if total_pages_display else page_to_fetch + 10,
                 value=st.session_state.current_page_number,
                 step=1,
                 key="page_selector"
            )
            if new_page_num != prev_page_number:
                st.session_state.current_page_number = new_page_num
                st.session_state.chat_history = []
                st.toast("Page changed, chat history cleared.", icon="ℹ️")
                st.rerun()

        with pcol2:
             if commits_on_current_page:
                 start_commit_num = (page_to_fetch - 1) * COMMITS_PER_PAGE + 1
                 end_commit_num = start_commit_num + len(commits_on_current_page) - 1
                 st.caption(f"Showing commits {start_commit_num} - {end_commit_num}")
             else:
                 st.caption("...")

    elif commits_on_current_page:
         st.subheader(f"Commits ({len(commits_on_current_page)} total)")

    # --- Display Summaries Section ---
    if error: st.error(f"Failed to fetch commits for page {page_to_fetch}: {error}")
    elif not commits_on_current_page and page_to_fetch == 1: st.info(f"No commits found matching your criteria.")
    elif not commits_on_current_page: st.info(f"No commits found on page {page_to_fetch}.")
    else:
        with st.expander(f"View Commit Summaries (Page {page_to_fetch})", expanded=True):
            st.markdown(f"**Summaries below are in {selected_language}:**")
            progress_bar = st.progress(0, text="Generating summaries...")
            summaries_container = st.container(height=600)
            total_on_page = len(commits_on_current_page)

            for i, commit in enumerate(commits_on_current_page):
                if 'summarize_commit' in globals(): summary = summarize_commit(commit, selected_language)
                else: summary = "Error: summarize_commit function not found."

                with summaries_container:
                    sha = commit.get('sha', 'N/A')
                    commit_info = commit.get('commit', {})
                    message = commit_info.get('message', 'N/A')
                    commit_header = f"[{sha[:7]}] {message.split(chr(10))[0][:80]}..."
                    st.markdown(f"#### {commit_header}")
                    st.markdown(f"**Summary ({selected_language}):**"); st.markdown(summary); st.markdown("---")
                    author_info = commit.get('commit', {}).get('author', {}); author_name = author_info.get('name', 'N/A') # Corrected author_info source
                    date_str = author_info.get('date', 'N/A'); html_url = commit.get('html_url')
                    col_details1, col_details2 = st.columns(2)
                    with col_details1:
                        st.caption(f"**Author:** {author_name}")
                        st.caption(f"**SHA:** {sha}")
                    # --- CORRECTED ---
                    with col_details2:
                        st.caption(f"**Date:** {date_str}")
                        # Use a standard 'if' statement for the conditional button
                        if html_url:
                            st.link_button("View on GitHub 🔗", html_url)
                    # --- END CORRECTION ---

                    show_full_message = st.toggle("Show Full Commit Message", key=f"msg_toggle_{sha}", value=False)
                    if show_full_message: st.code(message, language='text')
                    st.markdown("---")

                    show_diff = st.toggle("Show Code Changes", key=f"diff_toggle_{sha}", value=False)
                    if show_diff:
                        diff_placeholder = st.empty(); details, detail_error = None, None
                        with st.spinner("Fetching code changes..."): details, detail_error = get_commit_details(owner, repo, GITHUB_TOKEN, sha)
                        if detail_error: diff_placeholder.error(f"Could not load diff: {detail_error}")
                        elif details and 'files' in details and details['files']:
                            diff_placeholder.empty(); st.markdown("**Code Changes:**"); diff_found = False
                            for file in details.get('files', []):
                                filename = file.get('filename', 'Unknown file'); patch = file.get('patch'); status = file.get('status', '')
                                st.caption(f"`{filename}` ({status})")
                                if patch: st.code(patch, language='diff', line_numbers=True); diff_found = True
                                else: st.caption("_No textual diff available_")
                                st.divider()
                            if not diff_found: st.info("No textual diffs found.")
                        elif details and not details.get('files'): diff_placeholder.info("No files reported changed.")
                        else: diff_placeholder.warning("Could not retrieve valid diff details.")
                    st.divider() # Commit separator

                progress_bar.progress((i + 1) / total_on_page, text=f"Generating summaries... {i+1}/{total_on_page}")
            progress_bar.empty()

        # --- Chatbot Q&A Section ---
        st.markdown("---")
        st.subheader(f"❓ Ask Questions About Commits on Page {page_to_fetch}")
        st.markdown("_AI remembers the conversation since the page/filters changed._")
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]): st.markdown(message["content"])
        user_question = st.chat_input("Ask your question here...", key="qa_input")
        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with chat_container:
                with st.chat_message("user"): st.markdown(user_question)
            with st.spinner("Thinking..."):
                if 'answer_question' in globals() and 'format_commit_context' in globals():
                     commit_context = format_commit_context(commits_on_current_page)
                     ai_answer = answer_question(user_question, commit_context, st.session_state.chat_history)
                else: ai_answer = "Error: Q&A functions not found."
                st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})
            with chat_container:
                 with st.chat_message("assistant"): st.markdown(ai_answer)

elif not st.session_state.process_request and not st.session_state.repo_url:
     st.info("Enter a GitHub repository URL above and click 'Get Commits' to begin.")

# Add footer
st.divider()
st.caption(f"Current Date: {datetime.now(timezone.utc).date().isoformat()} (UTC)")
