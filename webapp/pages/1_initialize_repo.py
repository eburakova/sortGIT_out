import os
import google.generativeai as genai
from src.parse_git_log import parse_git_log_with_diff
from src.nlp import summarize_commit_info
import json
from dotenv import load_dotenv
from src.utils import get_git_log_with_diff
import streamlit as st

load_dotenv()

st.set_page_config(page_title="Initialize a repository", page_icon="!")

st.title("Initialize a repository")
st.write("""Create a knowledge base for the new repository.

Warning: uses Gemini API. 
""")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

DEFAULT_REPO_NAME = "ChefTreffHackFIChallenge"
APP_DIR= "."

path_to_repo = st.text_input("Type the repository name", placeholder=DEFAULT_REPO_NAME, value=DEFAULT_REPO_NAME)
repo_name = path_to_repo.split("/")[-1]

process_git_history = st.button("Process git history")
if process_git_history:
    gitlog_out = get_git_log_with_diff(f"../../{repo_name}")

# Write the raw output
    with open(f'{APP_DIR}/data/{repo_name}_git.log', 'w') as f:
        f.write(gitlog_out)

# Parse and create a json database
    with st.status("Processing repository...", expanded=True) as status:
        # Do some work
        st.markdown("**Extracting commit history from the repository...**")
        commit_data = parse_git_log_with_diff(gitlog_out)
        st.markdown(f"**Summarizing the commit history with AI...**")
        # More work
        commit_data = summarize_commit_info(commit_data, f'{APP_DIR}/data/{repo_name}_commit_DB.json')
        # Final step
        status.update(label="Processing complete!", state="complete")


# Summarize the differences and commit messages - LLM REQUESTS HERE!

    ### BACKUP
    with open(f'{APP_DIR}/data/{repo_name}_commit_DB.json', 'w') as f:
        json.dump(commit_data, f, indent=2)
    #st.info("Git history for the repository has been processed.")
    st.markdown(f"**The database is saved as** `{APP_DIR}/data/{repo_name}_commit_DB.json`")

# CREATE HIGHER-LEVEL SUMMARY

## Embeddings
#make_embeddings = st.button("Create embeddings?")

#if make_embeddings:




