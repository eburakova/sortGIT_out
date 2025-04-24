import streamlit as st
import requests
import os
import google.generativeai as genai

# ---- CONFIG ----
GEMINI_MODEL = "gemini-2.0-flash"  # or gemini-pro, if you prefer
GITHUB_API_BASE = "https://api.github.com/repos"

# ---- AUTH ----
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]  # Use Streamlit secrets for security

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

# ---- FUNCTIONS ----
def get_commits(repo_url, max_commits=200):
    try:
        parts = repo_url.rstrip("/").split("/")
        owner, repo = parts[-2], parts[-1]
        api_url = f"{GITHUB_API_BASE}/{owner}/{repo}/commits"
        res = requests.get(api_url)
        res.raise_for_status()
        return res.json()[:max_commits], None
    except Exception as e:
        return None, str(e)

def summarize_commit(commit):
    commit_message = commit['commit']['message']
    author = commit['commit']['author']['name']
    date = commit['commit']['author']['date']

#specified prompt
    prompt = f"""
You are an AI code assistant.
Summarize the following Git commit in simple, non-technical language:

Commit Message: {commit_message}
Author: {author}
Date: {date}

Make sure the summary is understandable by someone with no programming background.
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# ---- UI ----
st.title("GitHub Commit Summarizer (AI-Powered)")
repo_url = st.text_input("Paste a GitHub Repository URL:", placeholder="https://github.com/user/repo")

if repo_url:
    with st.spinner("Fetching commits..."):
        commits, error = get_commits(repo_url)
        if error:
            st.error(f"Failed to fetch commits: {error}")
        else:
            st.success(f"Found {len(commits)} commits")
            for commit in commits:
                summary = summarize_commit(commit)
                with st.expander(commit['commit']['message'][:60]):
                    st.markdown(summary)
                    st.caption(f"By {commit['commit']['author']['name']} on {commit['commit']['author']['date']}")
