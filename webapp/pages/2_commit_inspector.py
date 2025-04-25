import streamlit as st
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your utility functions
from src.utils import load_commit_data, format_git_time

st.set_page_config(page_title="Commit Inspector", layout="wide")

st.title("🔍 Commit Inspector")
st.markdown("Browse and inspect commits in your Git repository.")

# Repository selection
repo_name = st.sidebar.text_input("Repository name", value="linux_minidump")
json_path = f"data/{repo_name}_commit_DB.json"

# Load commit data
try:
    commit_data = load_commit_data(json_path)
    st.success(f"Loaded {len(commit_data)} commits for {repo_name}")
except FileNotFoundError:
    st.error(f"File not found: {json_path}")
    st.write(f"Current working directory: {os.getcwd()}")
    st.stop()

# Create a searchable/filterable table of commits
st.subheader("Commit History")

# Add filters
col1, col2, col3 = st.columns(3)
with col1:
    search_term = st.text_input("Search commits", placeholder="Search by message, author, or files")
with col2:
    start_date = st.date_input("Start date", value=None)
with col3:
    end_date = st.date_input("End date", value=None)
if start_date and end_date:
    if start_date <= end_date:
        st.success(f"Selected date range: {start_date} to {end_date}")
        try:
            filtered_commits_by_date = [commit for commit in commit_data if
                                    format_git_time(commit['date']) >= start_date and format_git_time(commit['date']) <= end_date]
            commit_data = filtered_commits_by_date
        except TypeError as e:
            st.error(f"Error: malformatted dates, can't filter {commit_data[0]['date']}, {e}")
    else:
        st.error("Error: End date must be after start date")

# Display commits in a table
if commit_data:
    # Create filtered data for display
    filtered_commits = []
    for commit in commit_data:
        for file in commit['files_changed']:
            if 'diff' in file:
                del file['diff']
    for commit in commit_data:
        # Apply search filter if provided
        if search_term and (search_term.lower() not in json.dumps(commit).lower()):
            continue
        overall_summaries = [
            file['diff_summary']['overall_summary']
            for file in commit['files_changed']
         if 'diff_summary' in file and 'overall_summary' in file['diff_summary']
        ]
        # Add to filtered list
        filtered_commits.append({
            "ID": commit["commit"][:7],
            "Author": commit["author"],
            "Date": commit["date"],
            "Commit msg": commit["message"],
            'Codechange summary': ' \n'.join(overall_summaries),
            #"Summary": commit.get("file_changes", [{"diff_summary": {"overall_summary": "No code based summary"}}])[0]['diff_summary'].get("overall_summary", 'No code-based summary') if commit.get("file_changed", False) else commit.get("summary", "No summary available"),
        })
        try:
            st.write(commit['file_changes'])
        except:
            pass

    #st.write(filtered_commits)
    # Display the commits
    st.write(f"Showing {len(filtered_commits)} of {len(commit_data)} commits")
    st.dataframe(filtered_commits)

    # Commit details section
    st.subheader("Commit Details")
    selected_hash = st.selectbox("Select commit to view details",
                                 options=[c["ID"] for c in filtered_commits])

    # Find and display the selected commit details
    for commit in commit_data:
        if commit["commit"].startswith(selected_hash):
            st.json(commit)
            break