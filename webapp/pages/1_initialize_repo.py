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

DEFAULT_PATH_TO_ROOT = "/ChefTreffHackFIChallenge"
APP_DIR= "/sortGIT_out/webapp/"

path_to_repo = st.text_input("Type the repository name", placeholder="ChefTreffHackFIChallenge")
repo_name = DEFAULT_PATH_TO_ROOT.split("/")[-1]
log_output = get_git_log_with_diff(DEFAULT_PATH_TO_ROOT)

# Write the raw output
with open(f'{APP_DIR}/data/{repo_name}_git.log', 'w') as f:
    f.write(log_output)

# Parse and create a json database
commit_data = parse_git_log_with_diff(log_output)
st.info(f"Summarizing the commit history with AI")

# Summarize the differences and commit messages - LLM REQUESTS HERE!
commit_data = summarize_commit_info(commit_data, f'{APP_DIR}/data/{repo_name}_commit_DB.json')

### BACKUP
with open(f'{APP_DIR}/data/{repo_name}_commit_DB.json', 'w') as f:
    json.dump(commit_data, f, indent=2)
st.info("Git history for the repository has been processed.")
st.markdown(f"The database is saved as {APP_DIR}/data/{repo_name}_commit_DB.json")


# CREATE HIGHER-LEVEL SUMMARY



