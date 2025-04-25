import subprocess
from src.parse_git_log import parse_git_log_with_diff
import os
import google.generativeai as genai
#from nlp import summarize_difference, summarize_commit_message
from src.nlp import summarize_commit_info
import logging
import json
from dotenv import load_dotenv
from src.utils import get_git_log_with_diff

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

lg = logging.getLogger(__name__)

PATH_TO_ROOT = '../../linux'
repo_name = PATH_TO_ROOT.split("/")[-1]
log_output = get_git_log_with_diff(PATH_TO_ROOT)

# Write the raw output
with open(f'../data/{repo_name}_git.log', 'w') as f:
    f.write(log_output)

# Parse and create a json database
commit_data = parse_git_log_with_diff(log_output)
lg.info("Summarizing the commit data with AI")
print(f"Summarizing the commit data with AI")

# Summarize the differences and commit messages - LLM REQUESTS HERE!
commit_data = summarize_commit_info(commit_data, f'../data/{repo_name}_commit_DB.json')

### BACKUP
with open(f'../data/{repo_name}_commit_DB.json', 'w') as f:
    #f.write(str(commit_data))
    json.dump(commit_data, f, indent=2)

# CREATE HIGHER-LEVEL SUMMARY



