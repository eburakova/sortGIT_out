import streamlit as st
import requests
import os
import google.generativeai as genai
from datetime import datetime, date, timedelta  # Added for date filtering

# ---- CONFIG ----
GEMINI_MODEL = "gemini-2.0-flash"  # Or your preferred model like gemini-1.5-flash
GITHUB_API_BASE = "https://api.github.com/repos"
# Define supported languages for the UI and translation prompt
SUPPORTED_LANGUAGES = {
    "English": "English",
    "German": "German",
    "French": "French",
    "Spanish": "Spanish",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "Mandarin Chinese": "Mandarin Chinese"
}


# ---- AUTH ----
# Make sure to set GEMINI_API_KEY in Streamlit secrets ([secrets.toml])
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
except (KeyError, Exception) as e:
    st.error(f"Error configuring Gemini API: {e}. Please ensure GEMINI_API_KEY is set in Streamlit secrets.")
    st.stop() # Stop execution if API key is missing


# ---- FUNCTIONS ----
@st.cache_data(ttl=3600) # Cache GitHub API calls for 1 hour
def get_commits(repo_url, max_commits=200):
    """Fetches commits from a GitHub repository."""
    try:
        # Basic validation and parsing
        if not repo_url.startswith("https://github.com/"):
             return None, "Invalid GitHub repository URL format. Must start with https://github.com/"
        parts = repo_url.strip("/").split("/")
        if len(parts) < 5: # e.g. https://github.com/owner/repo
             return None, "Invalid GitHub repository URL format. Missing owner or repo name."
        owner, repo = parts[-2], parts[-1]

        # Construct API URL and fetch commits
        api_url = f"{GITHUB_API_BASE}/{owner}/{repo}/commits"
        params = {'per_page': max_commits} # Fetch up to max_commits in one go
        res = requests.get(api_url, params=params, timeout=15) # Added timeout
        res.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return res.json(), None
    except requests.exceptions.RequestException as e:
        return None, f"Network or GitHub API error: {e}"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"

# Cache summaries to avoid redundant API calls, especially if filtering changes
@st.cache_data(ttl=86400) # Cache summaries for 1 day
def summarize_commit(_commit_data, target_language="English"):
    """Summarizes a commit using the Gemini model and translates if needed."""
    commit_message = _commit_data['commit']['message']
    author = _commit_data['commit']['author']['name']
    date_str = _commit_data['commit']['author']['date']
    sha = _commit_data['sha']

    # Base prompt
    prompt = f"""
You are an expert AI assistant specializing in code analysis.
Analyze and summarize the following Git commit in simple, non-technical language.
Focus on the *what* and *why* of the change, understandable by someone with minimal programming background.

Commit SHA: {sha}
Commit Message: {commit_message}
Author: {author}
Date: {date_str}
"""

    # Add translation instruction if the target language is not English
    if target_language != "English":
        prompt += f"\n\nPlease provide the summary exclusively in {target_language}."

    try:
        response = model.generate_content(prompt)
        # Handle potential safety blocks or empty responses
        if not response.parts:
             return f"Could not generate summary for commit {sha[:7]}. The response was empty or blocked."
        return response.text
    except Exception as e:
        # Log the error for debugging if needed (e.g., print(f"Gemini API Error: {e}"))
        return f"Error generating summary for commit {sha[:7]}: {e}"

def parse_commit_date(date_string):
    """Parses GitHub's ISO 8601 date string into a date object."""
    try:
        # Handle the 'Z' for UTC timezone
        if date_string.endswith('Z'):
            date_string = date_string[:-1] + '+00:00'
        return datetime.fromisoformat(date_string).date()
    except ValueError:
        # Fallback or logging if needed
        return None

# ---- UI ----
st.set_page_config(layout="wide") # Use wider layout
st.title("🚀 AI-Powered GitHub Commit Summarizer")
st.markdown("Get easy-to-understand summaries of commits from any public GitHub repository.")

# --- Input Area ---
col1, col2 = st.columns([3, 1]) # Give more space to URL input
with col1:
    repo_url = st.text_input(
        "Paste a Public GitHub Repository URL:",
        placeholder="e.g., https://github.com/streamlit/streamlit",
        key="repo_url_input"
    )
with col2:
    # Language Selection
    selected_language = st.selectbox(
        "Select Summary Language:",
        options=list(SUPPORTED_LANGUAGES.keys()),
        index=0, # Default to English
        key="language_select"
    )

# --- Filtering Area ---
st.subheader("🔎 Filter Commits (Optional)")
filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    # Specific Commit SHA Filter
    specific_commit_sha = st.text_input(
        "Filter by Commit SHA (or prefix):",
        placeholder="e.g., a1b2c3d",
        key="sha_filter"
    ).strip().lower() # Normalize input

with filter_col2:
    # Date Range Filter - Start Date
    # Default to None initially to avoid filtering unless user selects a date
    start_date = st.date_input(
        "Filter From Date:",
        value=None, # No default start date
        key="start_date_filter"
    )

with filter_col3:
    # Date Range Filter - End Date
    # Default to today
    end_date = st.date_input(
        "Filter To Date:",
        value="today", # Default to today's date
        key="end_date_filter"
    )

# Add a button to trigger processing
process_button = st.button("Get & Summarize Commits", key="process_button")

# --- Main Processing Logic ---
if process_button and repo_url:
    with st.spinner(f"Fetching commits from {repo_url}... Please wait."):
        commits_data, error = get_commits(repo_url)

    if error:
        st.error(f"Failed to fetch commits: {error}")
    elif not commits_data:
        st.warning("No commits found for this repository or the repository is empty.")
    else:
        st.success(f"Fetched {len(commits_data)} latest commits.")

        # Filter commits based on UI inputs
        filtered_commits = commits_data
        original_count = len(filtered_commits)

        # 1. Filter by Specific SHA
        if specific_commit_sha:
            filtered_commits = [
                c for c in filtered_commits
                if c['sha'].lower().startswith(specific_commit_sha)
            ]
            st.info(f"Filtered by SHA prefix '{specific_commit_sha}': {len(filtered_commits)} commits remain.")

        # 2. Filter by Date Range
        if start_date or end_date: # Only filter if at least one date is set
            filtered_commits_date = []
            # Set effective start/end dates for comparison
            effective_start_date = start_date if start_date else date.min # If no start date, use earliest possible date
            effective_end_date = end_date if end_date else date.today() # If no end date, use today

            for commit in filtered_commits:
                commit_date_obj = parse_commit_date(commit['commit']['author']['date'])
                if commit_date_obj and effective_start_date <= commit_date_obj <= effective_end_date:
                     filtered_commits_date.append(commit)

            # Show info only if date filtering actually changed the list
            if len(filtered_commits_date) != len(filtered_commits):
                 st.info(f"Filtered by date range ({effective_start_date} to {effective_end_date}): {len(filtered_commits_date)} commits remain.")
            filtered_commits = filtered_commits_date


        # Check if any commits remain after filtering
        if not filtered_commits:
            st.warning("No commits match your filter criteria.")
        else:
            st.markdown(f"**Displaying summaries for {len(filtered_commits)} filtered commits (in {selected_language}):**")

            # Display summaries for filtered commits
            # Use st.expander for each commit summary
            progress_bar = st.progress(0)
            total_filtered = len(filtered_commits)
            summaries_container = st.container() # Use a container to add summaries progressively

            for i, commit in enumerate(filtered_commits):
                # Pass the commit data (as a dictionary) and language to the caching function
                # Note: Caching works based on the input arguments. If _commit_data or target_language changes, it recalculates.
                commit_tuple = tuple(sorted(commit.items())) # Convert dict to tuple for caching if needed, but dicts often work if stable
                summary = summarize_commit(commit, selected_language) # Pass selected language

                with summaries_container:
                    # Display commit info and summary
                    commit_sha_short = commit['sha'][:7] # Short SHA
                    commit_msg_short = commit['commit']['message'].split('\n')[0][:80] # First line, truncated
                    expander_title = f"[{commit_sha_short}] {commit_msg_short}..."

                    with st.expander(expander_title):
                        st.markdown(f"**Summary ({selected_language}):**")
                        st.markdown(summary) # Display the summary text from Gemini
                        st.markdown("---")
                        st.caption(f"**Full Message:** {commit['commit']['message']}")
                        st.caption(f"**Author:** {commit['commit']['author']['name']}")
                        st.caption(f"**Date:** {commit['commit']['author']['date']}")
                        st.caption(f"**SHA:** {commit['sha']}")
                        # Add a link to the commit on GitHub
                        commit_url = commit.get('html_url')
                        if commit_url:
                            st.markdown(f"[View on GitHub]({commit_url})", unsafe_allow_html=True)

                # Update progress bar
                progress_bar.progress((i + 1) / total_filtered)
            progress_bar.empty() # Remove progress bar after completion
elif process_button and not repo_url:
    st.warning("Please enter a GitHub repository URL.")
