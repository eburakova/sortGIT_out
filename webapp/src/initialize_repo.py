import subprocess
from parse_git_log import parse_git_log_with_diff
import os
import google.generativeai as genai
from nlp import summarize_difference, summarize_commit_message
import logging
import json
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

lg = logging.getLogger(__name__)

def get_git_log_with_diff(PATH_TO_ROOT):
    result = subprocess.run(
        ['git', 'log', '-p', '--no-color'],
        cwd=PATH_TO_ROOT,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


PATH_TO_ROOT = '../../../commit-messages-guide'
repo_name = PATH_TO_ROOT.split("/")[-1]
log_output = get_git_log_with_diff(PATH_TO_ROOT)

# Write the raw output
with open(f'../data/{repo_name}_git.log', 'w') as f:
    f.write(log_output)

# Parse and create a json database
commit_data = parse_git_log_with_diff(log_output)
lg.info("Summarizing the commit data with AI")
print(f"Summarizing the commit data with AI")

# Summarize the differences and commit messages
for j, commit in enumerate(commit_data):
    commit['summary'] = summarize_commit_message(commit['message'])
    with open(f'../data/{repo_name}_commit_DB.json', 'w') as f:
        json.dump(commit_data, f, indent=2)
    for i, _ in enumerate(commit['files_changed']):
        try:
            diff_message = commit['files_changed'][i]['diff']
            commit['files_changed'][i]['diff_summary'] = summarize_difference(diff_message)
            lg.info(f"Commit processed: {commit['message']}")
            print(f"Commit processed: {commit['commit']} \t {commit['message']}")
        except Exception as e:
            print(e)
            commit['files_changed'][i]['diff_summary'] = "No summary available"


with open(f'../data/{repo_name}_commit_DB.json', 'w') as f:
    #f.write(str(commit_data))
    json.dump(commit_data, f, indent=2)


