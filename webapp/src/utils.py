import subprocess
import streamlit as st
import json
from datetime import datetime

def get_git_log_with_diff(PATH_TO_ROOT):
    result = subprocess.run(
        ['git', 'log', '-p', '--no-color'],
        cwd=PATH_TO_ROOT,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout

@st.cache_data
def load_commit_data(filepath: str):
    """
    Load commit metadata JSON from disk and cache the result.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def format_commit_data(commits):
    lines=["Here is a database of Git commits:\n"]
    for c in commits:
        time_str = c['date']
        # Parse the string into a datetime object
        dt = datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y %z")
        ts_formatted = dt.strftime("%Y-%m-%d %H:%M:%S %z")
        try:
            files = ', '.join(c.get('files_changed', []))
        except TypeError:
            try:
                files = ', '.join(c.get('files_changed', {}))
            except Exception:
                files = str(c['files_changed'])
        prompt = f"- Commit `{c['commit']}` by {c['author']} on {ts_formatted}"
        if 'summary' in c.keys():
            prompt += f"{c['summary']}"
        prompt += "\n" + files
        lines.append(prompt)
    return "\n".join(lines)

def data_sanity_check(init_prompt):
    st.write("Data is too large for an LLM! Truncating...")
    return