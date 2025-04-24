import streamlit as st
import requests
import os
import google.generativeai as genai
from datetime import datetime, date, timedelta
import re # Import regex for parsing repo URL

# ---- CONFIG ----
GEMINI_MODEL = "gemini-1.5-flash"
GITHUB_API_BASE = "https://api.github.com/repos"
MAX_COMMITS_FOR_CONTEXT = 50
MAX_HISTORY_TURNS = 5
# Define supported languages for the UI and translation prompt
SUPPORTED_LANGUAGES = {
    "English": "English", "German": "German", "French": "French",
    "Spanish": "Spanish", "Japanese": "Japanese", "Korean": "Korean",
    "Mandarin Chinese": "Mandarin Chinese"
}

# ---- AUTH ----
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
except (KeyError, AttributeError):
    st.error("GEMINI_API_KEY not found. Please add it to your Streamlit secrets.")
    st.info("Refer to Streamlit docs: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management")
    st.stop()
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# ---- HELPER FUNCTION ----
def parse_repo_url(url):
    """Parses GitHub URL to extract owner and repo."""
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1), match.group(2).replace('.git', '') # Owner, Repo
    return None, None

# ---- API FUNCTIONS ----
@st.cache_data(ttl=3600) # Cache list for 1 hour
def get_commits(owner, repo, max_commits=200):
    """Fetches commits list for a repository."""
    if not owner or not repo:
        return None, "Invalid repository owner or name."
    api_url = f"{GITHUB_API_BASE}/{owner}/{repo}/commits"
    params = {'per_page': min(max_commits, 100) if max_commits > 0 else 100}
    all_commits = []
    page = 1
    target_count = max_commits if max_commits > 0 else float('inf')
    try:
        while len(all_commits) < target_count:
            params['page'] = page
            res = requests.get(api_url, params=params, timeout=20)
            res.raise_for_status()
            fetched_commits = res.json()
            if not isinstance(fetched_commits, list): return None, f"Unexpected API response: {type(fetched_commits)}"
            if not fetched_commits: break
            all_commits.extend(fetched_commits)
            if len(fetched_commits) < params['per_page']: break
            page += 1
        return all_commits[:max_commits] if max_commits > 0 else all_commits, None
    except requests.exceptions.RequestException as e:
        error_message = f"API error fetching commit list: {e}"
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code == 404: error_message = f"Repository '{owner}/{repo}' not found. Check URL."
            elif e.response.status_code == 403: error_message = f"Access forbidden fetching list (Private repo or rate limit?)."
            else: error_message += f" (Status code: {e.response.status_code})"
        return None, error_message
    except Exception as e: return None, f"Unexpected error fetching commit list: {e}"

# --- NEW FUNCTION ---
@st.cache_data(ttl=86400) # Cache individual commit details for 1 day
def get_commit_details(owner, repo, commit_sha):
    """Fetches detailed information for a single commit, including file diffs."""
    if not owner or not repo or not commit_sha:
        return None, "Missing owner, repo, or SHA for fetching commit details."
    api_url = f"{GITHUB_API_BASE}/{owner}/{repo}/commits/{commit_sha}"
    try:
        res = requests.get(api_url, timeout=15)
        res.raise_for_status()
        commit_data = res.json()
        # Check if 'files' key exists, indicating diff info is likely present
        if 'files' in commit_data:
            return commit_data, None
        else:
            return None, "Commit data fetched, but file change details ('files' key) are missing."
    except requests.exceptions.RequestException as e:
        error_message = f"API error fetching commit details for {commit_sha[:7]}: {e}"
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code == 404: error_message = f"Commit {commit_sha} not found."
            elif e.response.status_code == 403: error_message = f"Access forbidden fetching details (Rate limit?)."
            # Handle 422 Unprocessable Entity - common if commit history is too complex/large for diff
            elif e.response.status_code == 422: error_message = f"Cannot retrieve diff for commit {commit_sha[:7]}. May be too large or complex."
            else: error_message += f" (Status code: {e.response.status_code})"
        return None, error_message
    except Exception as e:
        return None, f"Unexpected error fetching details for commit {commit_sha[:7]}: {e}"

# --- Summarization and other functions remain the same ---
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
             return f"Could not generate summary for commit {sha[:7]}. Reason: Blocked due to {reason} content."
        else: return f"Could not generate summary for commit {sha[:7]}. Received an empty or invalid response."
    except Exception as e: return f"Error generating summary for commit {sha[:7]}: API or processing error."

def parse_commit_date(date_string):
    """Parses GitHub's ISO 8601 date string into a date object."""
    if not date_string or not isinstance(date_string, str): return None
    try:
        if date_string.endswith('Z'): date_string = date_string[:-1] + '+00:00'
        return datetime.fromisoformat(date_string).date()
    except (ValueError, TypeError): return None

def format_commit_context(commits):
    """Formats commit data into a string context for the chatbot."""
    if not commits: return "No commit data available."
    context_str = "Relevant commit data:\n\n"
    commits_to_include = commits[:MAX_COMMITS_FOR_CONTEXT]
    for commit in commits_to_include:
        sha = commit.get('sha', 'N/A')
        commit_info = commit.get('commit', {})
        author_info = commit_info.get('author', {})
        author = author_info.get('name', 'N/A')
        date_str = author_info.get('date', 'N/A')
        message = commit_info.get('message', 'N/A')
        context_str += f"Commit SHA: {sha}\nAuthor: {author}\nDate: {date_str}\nMessage:\n{message}\n---\n"
    if len(commits) > MAX_COMMITS_FOR_CONTEXT:
         context_str += f"\nNote: Context includes details for the first {MAX_COMMITS_FOR_CONTEXT} of {len(commits)} filtered commits."
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
    if not commit_context or commit_context == "No commit data available.": return "Cannot answer questions without commit data."
    formatted_chat_history = format_chat_history(chat_history_list)
    prompt = f"""
You are a helpful AI assistant acting as a communicator who translates technical Git commit information for a **non-technical audience**.
Your goal is to answer the user's **latest question** based *only* on the **Provided Commit Data** below, considering the **Ongoing Conversation History** for context. Explain the answer in **simple, everyday language**.

**Instructions for your answer:**
* Analyze the **Provided Commit Data**. This is your *primary source* for facts about the commits.
* Review the **Ongoing Conversation History** to understand the context of the user's latest question.
* Answer the **User's Latest Question** accurately based *only* on the commit data.
* **Explain your answer clearly and concisely using non-technical language.** Avoid jargon. Focus on 'what' and 'why'.
* Use the conversation history to resolve ambiguities if possible, but **do not base factual answers on the history itself, only on the commit data.**
* If technical terms are unavoidable, briefly explain them simply.
* Imagine you are explaining to a project manager or client.
* If the answer cannot be found in the commit data, state that clearly.

**Provided Commit Data:**
{commit_context}
---
**Ongoing Conversation History:**
{formatted_chat_history}
---
**User's Latest Question:**
{question}
---
**Answer (explained in simple, non-technical terms, considering conversation history):**
"""
    try:
        response = model.generate_content(prompt)
        if response.parts: return response.text
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             reason = response.prompt_feedback.block_reason.name.replace("_", " ").lower()
             return f"Sorry, I couldn't generate an answer. Blocked due to {reason} content."
        else: return "Sorry, I received an empty response from the AI."
    except Exception as e: return f"Sorry, an error occurred while trying to generate the answer: {e}"

# ---- Session State Initialization ----
if 'filtered_commits' not in st.session_state: st.session_state.filtered_commits = []
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'repo_url' not in st.session_state: st.session_state.repo_url = ""
# Store owner/repo in session state once parsed
if 'owner' not in st.session_state: st.session_state.owner = None
if 'repo' not in st.session_state: st.session_state.repo = None


# ---- UI ----
st.set_page_config(layout="wide", page_title="AI Commit Summarizer & Q&A")
st.title("🚀 AI-Powered GitHub Commit Summarizer & Q&A")
st.markdown("Get summaries and ask questions about commits from a public GitHub repository.")

# --- Input Area ---
col1, col2 = st.columns([3, 1])
with col1:
    st.session_state.repo_url = st.text_input(
        "Paste a Public GitHub Repository URL:", placeholder="e.g., https://github.com/streamlit/streamlit",
        key="repo_url_input", value=st.session_state.repo_url,
        help="Enter the full URL of the public GitHub repository."
    )
    repo_url = st.session_state.repo_url
    # Parse owner/repo here and store in session state if URL is valid
    if repo_url:
        owner, repo = parse_repo_url(repo_url)
        if owner and repo:
            st.session_state.owner = owner
            st.session_state.repo = repo
        else:
            # Clear owner/repo if URL becomes invalid
            st.session_state.owner = None
            st.session_state.repo = None
            st.warning("Invalid GitHub URL format.", icon="⚠️")

with col2:
    selected_language = st.selectbox(
        "Select Summary Language:", options=list(SUPPORTED_LANGUAGES.keys()), index=0, key="language_select",
        help="Choose the language for the AI-generated commit summaries."
    )

# --- Filtering Area ---
# (Filtering UI remains the same)
st.subheader("🔎 Filter Commits (Optional)")
filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    specific_commit_sha = st.text_input("Filter by Commit SHA (prefix):", placeholder="e.g., a1b2c3d", key="sha_filter", help="Enter starting characters of a commit SHA.").strip().lower()
with filter_col2:
    start_date = st.date_input("Filter From Date:", value=None, key="start_date_filter", help="Show commits from this date.")
with filter_col3:
    today_date = datetime.now().date()
    end_date = st.date_input("Filter To Date:", value=today_date, key="end_date_filter", min_value=start_date if start_date else None, max_value=today_date, help="Show commits up to this date.")
if start_date and end_date and start_date > end_date: st.warning("Start date cannot be after end date.")

# --- Process Button ---
process_button = st.button("Get & Summarize Commits", key="process_button", type="primary", disabled=not st.session_state.owner) # Disable if owner/repo invalid

# --- Main Processing & Summary Display Logic ---
if process_button and st.session_state.owner and st.session_state.repo:
    if start_date and end_date and start_date > end_date: st.error("Invalid date range: Start date is after end date.")
    else:
        with st.spinner(f"Fetching commits from {st.session_state.owner}/{st.session_state.repo}..."):
            st.session_state.filtered_commits = []
            st.session_state.chat_history = []
            # Use owner/repo from session state
            commits_data, error = get_commits(st.session_state.owner, st.session_state.repo, max_commits=200)

        if error: st.error(f"Failed to fetch commits: {error}")
        elif not commits_data: st.warning("No commits found.")
        else:
            st.success(f"Fetched {len(commits_data)} latest commits. Applying filters...")
            # --- Filtering Logic (remains the same) ---
            filtered_commits_list = commits_data
            filtering_info = []
            if specific_commit_sha:
                count_before_sha = len(filtered_commits_list)
                filtered_commits_list = [c for c in filtered_commits_list if c.get('sha', '').lower().startswith(specific_commit_sha)]
                if len(filtered_commits_list) != count_before_sha: filtering_info.append(f"Filtered by SHA '{specific_commit_sha}'.")
            effective_start_date = start_date if start_date else date.min
            effective_end_date = end_date if end_date else today_date
            count_before_date = len(filtered_commits_list)
            temp_filtered_commits_date = [c for c in filtered_commits_list if (commit_date_obj := parse_commit_date(c.get('commit', {}).get('author', {}).get('date'))) and effective_start_date <= commit_date_obj <= effective_end_date]
            if len(temp_filtered_commits_date) != count_before_date: filtering_info.append(f"Filtered by date ({effective_start_date} to {effective_end_date}).")
            filtered_commits_list = temp_filtered_commits_date
            # --- End Filtering Logic ---

            st.session_state.filtered_commits = filtered_commits_list
            if filtering_info: st.info(" ".join(filtering_info) + f" Showing {len(st.session_state.filtered_commits)} commits.")
            elif len(commits_data)>0: st.info(f"No effective filters applied. Showing {len(st.session_state.filtered_commits)} commits.")

            # --- Display Summaries Section (Now inside an expander) ---
            if not st.session_state.filtered_commits:
                st.warning("No commits match your filter criteria.")
            else:
                # --- OUTER EXPANDER ---
                with st.expander(f"View Commit Summaries ({len(st.session_state.filtered_commits)} Commits)", expanded=True):
                    st.markdown(f"**Summaries below are in {selected_language}:**")
                    progress_bar = st.progress(0, text="Generating summaries...")
                    total_filtered = len(st.session_state.filtered_commits)
                    # Container for scrollable summaries
                    summaries_container = st.container(height=500) # Adjust height if needed

                    for i, commit in enumerate(st.session_state.filtered_commits):
                        # Get summary (uses cache)
                        summary = summarize_commit(commit, selected_language)

                        with summaries_container:
                            # --- INNER EXPANDER (per commit) ---
                            sha = commit.get('sha', 'N/A')
                            commit_info = commit.get('commit', {})
                            message = commit_info.get('message', 'N/A')
                            author_info = commit_info.get('author', {})
                            author_name = author_info.get('name', 'N/A')
                            date_str = author_info.get('date', 'N/A')
                            html_url = commit.get('html_url')
                            expander_title = f"[{sha[:7]}] {message.split(chr(10))[0][:80]}..."

                            with st.expander(expander_title):
                                st.markdown(f"**Summary ({selected_language}):**")
                                st.markdown(summary) # Display the AI summary
                                st.markdown("---")
                                # Display commit details
                                st.caption(f"**Full Message:**\n```\n{message}\n```")
                                st.caption(f"**Author:** {author_name}")
                                st.caption(f"**Date:** {date_str}")
                                st.caption(f"**SHA:** {sha}")
                                if html_url: st.markdown(f"🔗 [View on GitHub]({html_url})", unsafe_allow_html=True)

                                # --- Code Diff Toggle and Display ---
                                st.markdown("---") # Separator for diff section
                                # Use toggle for better UX than checkbox inside expander
                                show_diff = st.toggle("Show Code Changes", key=f"diff_toggle_{sha}")
                                if show_diff:
                                    # Fetch details only when toggle is True
                                    with st.spinner("Fetching code changes..."):
                                        # Get owner/repo from session state
                                        owner = st.session_state.owner
                                        repo = st.session_state.repo
                                        # Call the new function to get details
                                        details, detail_error = get_commit_details(owner, repo, sha)

                                    if detail_error:
                                        st.error(f"Could not load diff: {detail_error}")
                                    elif details and 'files' in details:
                                        st.markdown("**Code Changes:**")
                                        # Check if there are any files with patch data
                                        diff_found = False
                                        for file in details.get('files', []):
                                            filename = file.get('filename', 'Unknown file')
                                            patch = file.get('patch') # Diff content
                                            status = file.get('status', '') # e.g., 'added', 'modified', 'removed'

                                            # Display filename and status
                                            st.caption(f"`{filename}` ({status})")

                                            # Display the diff using st.code if patch exists
                                            if patch:
                                                st.code(patch, language='diff', line_numbers=True)
                                                diff_found = True
                                            else:
                                                # Indicate if patch data is missing (e.g., binary file, large diff not generated)
                                                st.caption("_No textual diff available for this file (might be binary or too large)._")
                                            st.markdown("---") # Separator between files
                                        if not diff_found and details.get('files'):
                                             st.info("No textual diffs were found for the changed files in this commit.")
                                        elif not details.get('files'):
                                             st.info("No files were reported as changed in this commit's details.")
                                    else:
                                        st.warning("Could not retrieve file change details for this commit.")
                                # --- End Code Diff Section ---

                        # Update progress bar after processing each commit
                        progress_bar.progress((i + 1) / total_filtered, text=f"Generating summaries... {i+1}/{total_filtered}")
                    # Clear progress bar after loop
                    progress_bar.empty()
                # --- END OUTER EXPANDER ---
elif process_button and not st.session_state.owner:
    st.warning("Please enter a valid GitHub repository URL first.")


# --- Chatbot Q&A Section ---
# Show only if commits are loaded (check filtered_commits list directly)
if st.session_state.filtered_commits: # Check if the list exists and is not empty
    st.markdown("---")
    st.subheader("❓ Ask Questions About These Commits")
    st.markdown("_AI remembers the conversation in this session. Ask follow-up questions!_")

    chat_container = st.container(height=400) # Make chat scrollable
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_question = st.chat_input("Ask your question here...", key="qa_input")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with chat_container: # Display immediately in the correct container
            with st.chat_message("user"):
                st.markdown(user_question)

        with st.spinner("Thinking..."):
            commit_context = format_commit_context(st.session_state.filtered_commits)
            ai_answer = answer_question(user_question, commit_context, st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})

            # Display AI answer - happens automatically on next Streamlit cycle after state update
            # Need to trigger rerun if using st.text_area, but not needed for st.chat_input usually
            # We can force it if needed: st.rerun()
        # This structure with chat_input usually updates the display correctly without explicit rerun

    # This ensures the latest assistant message is displayed even without rerun
    # Check if last message is assistant and display it if chat input was processed
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant" and user_question :
         with chat_container:
              with st.chat_message("assistant"):
                   st.markdown(st.session_state.chat_history[-1]["content"])


# Add footer

st.caption(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
