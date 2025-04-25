import streamlit as st
import os
import google.generativeai as genai
from datetime import datetime, date, timedelta, timezone
import re
import math
import git
from pathlib import Path
import traceback
import numpy as np # Added for vector operations
from scipy.spatial.distance import cosine # Added for similarity calculation

# ---- CONFIG ----
GENERATIVE_MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL_NAME = "models/text-embedding-004" # Or other available embedding model
MAX_HISTORY_TURNS = 5
GLOBAL_MAX_HISTORY_TURNS = 8 # Allow slightly more history for global chat
COMMITS_PER_PAGE = 30
# How many relevant commits to retrieve for global Q&A context
NUM_RELEVANT_COMMITS = 5

# Define supported languages for the UI and translation prompt
SUPPORTED_LANGUAGES = {
    "English": "English", "German": "German", "French": "French",
    "Spanish": "Spanish", "Japanese": "Japanese", "Korean": "Korean",
    "Mandarin Chinese": "Mandarin Chinese"
}

# ---- AUTH & MODEL INITIALIZATION ----
try:
    # Ensure you have the GEMINI_API_KEY in your Streamlit secrets
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY missing.")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    # Initialize embedding client (implicitly handled by genai.embed_content)
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

# validate_repo_path (remains the same)
def validate_repo_path(path_str):
    """Checks if the path exists and is a valid Git repository."""
    if not path_str:
        return False, "Repository path cannot be empty."
    try:
        path = Path(path_str).resolve()
        if not path.is_dir():
            return False, f"Path '{path_str}' is not a valid directory."
        try:
            _ = git.Repo(str(path)) # Attempt to instantiate Repo object
            return True, str(path) # Return the resolved path string if successful
        except git.InvalidGitRepositoryError:
            return False, f"Path '{str(path)}' does not appear to be a valid Git repository."
        except git.NoSuchPathError:
             return False, f"Path '{str(path)}' does not exist."
    except Exception as e:
        # Handle potential permission errors or other Path issues
        return False, f"Error validating path '{path_str}': {e}"

# get_local_commits (remains the same)
@st.cache_data(ttl=120)
def get_local_commits(_repo_path_str, sha_filter=None, start_date=None, end_date=None,
                       author_filter=None, committer_filter=None, message_filter=None,
                       path_filter=None, include_merges=False):
    """
    Fetches serializable commit data from the local repository using GitPython,
    applying all specified filters.
    Returns a list of dictionaries, not raw Commit objects.
    """
    is_valid, msg_or_path = validate_repo_path(_repo_path_str)
    if not is_valid: return None, msg_or_path
    serializable_commits = []
    try:
        repo = git.Repo(msg_or_path)
        repo.git.status() # Check if repo is usable
        iter_args = {'no_merges': not include_merges} # Default based on checkbox

        # --- Handle Branch/Tag/SHA Filter ---
        if sha_filter:
            try:
                # Resolve symbolic refs (branches/tags) to SHAs for robustness
                resolved_ref = repo.commit(sha_filter)
                iter_args['rev'] = resolved_ref.hexsha
            except (git.BadName, git.BadObject, ValueError) as e: # Catch specific GitPython errors
                 return None, f"Error: Branch/Tag/SHA filter '{sha_filter}' not found or invalid in the repository: {e}"
            except Exception as e: # Catch other potential errors during commit resolution
                 return None, f"Error resolving filter '{sha_filter}': {e}"
        else:
            # Default to HEAD if no specific revision filter is given
             try:
                 if not repo.references: return [], "Repository appears to be empty (no references/commits)."
                 # Handle detached HEAD or active branch
                 if repo.head.is_detached:
                     iter_args['rev'] = repo.head.commit.hexsha
                 else:
                     iter_args['rev'] = repo.active_branch.commit.hexsha
             except (ValueError, TypeError, AttributeError) as e: # Handles various states like empty repo, initial commit error etc.
                  return None, f"Repository HEAD seems invalid or empty: {e}"

        # --- Handle Date Filters ---
        if start_date:
            # Ensure datetime objects are timezone-aware for GitPython
            start_dt_aware = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
            iter_args['since'] = start_dt_aware
        if end_date:
            end_dt_aware = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc).replace(microsecond=0) # End of day
            iter_args['until'] = end_dt_aware

        # --- Handle Author/Committer Filters (passed directly) ---
        if author_filter:
            iter_args['author'] = author_filter
        if committer_filter:
            iter_args['committer'] = committer_filter

        # --- Handle Path Filter (passed directly) ---
        # Ensure path_filter is not empty string if provided
        if path_filter and path_filter.strip():
            # GitPython expects a list of paths, even if just one
            iter_args['paths'] = [path_filter.strip()]

        # --- Iterate and Extract Data ---
        commit_iterator = repo.iter_commits(**iter_args)
        temp_commits = [] # Store results before message filtering

        for commit_obj in commit_iterator:
            # Extract serializable data first
            commit_data = {
                'sha': commit_obj.hexsha,
                'message': commit_obj.message,
                'authored_datetime': commit_obj.authored_datetime.isoformat(),
                'committed_datetime': commit_obj.committed_datetime.isoformat(),
                'author_name': commit_obj.author.name,
                'author_email': commit_obj.author.email,
                'committer_name': commit_obj.committer.name,
                'committer_email': commit_obj.committer.email,
                'parents': [p.hexsha for p in commit_obj.parents],
                'stats': None, # Placeholder
                'files': None,  # Placeholder
            }
            temp_commits.append(commit_data) # Add to temporary list

        # --- Handle Message Filter (Post-filtering) ---
        if message_filter and message_filter.strip():
            pattern = re.compile(re.escape(message_filter.strip()), re.IGNORECASE | re.MULTILINE) # Case-insensitive search
            serializable_commits = [
                commit for commit in temp_commits
                if pattern.search(commit['message'])
            ]
        else:
            # If no message filter, use all commits found
            serializable_commits = temp_commits

        return serializable_commits, None # Return the final list of dictionaries

    except git.InvalidGitRepositoryError:
        return None, f"Error: Path '{msg_or_path}' is not a valid Git repository."
    except git.NoSuchPathError:
         return None, f"Error: Path '{msg_or_path}' does not exist."
    except git.GitCommandError as e:
        # Provide more specific error context if possible
        return None, f"Git command error during commit listing: {e.stderr or e}"
    except Exception as e:
        st.error(traceback.format_exc()) # Log full traceback for debugging
        return None, f"Unexpected error reading local repository: {e}"

# get_commit_diff (remains the same)
@st.cache_data(ttl=3600)
def get_commit_diff(_repo_path_str, commit_sha):
    """Gets the diff (patch) for a specific commit SHA."""
    is_valid, msg_or_path = validate_repo_path(_repo_path_str)
    if not is_valid:
        return None, f"Invalid repository path provided for diff: {msg_or_path}"
    if not commit_sha:
        return None, "Commit SHA must be provided to get diff."

    try:
        repo = git.Repo(msg_or_path)
        commit_obj = repo.commit(commit_sha) # Get the specific commit object
        diff_data = {'files': [], 'diff_error': None}

        # Handle initial commit separately
        if not commit_obj.parents:
             try:
                 # Use git show for the initial commit diff (basic parsing included)
                 raw_diff = repo.git.show(commit_obj.hexsha, pretty="", patch=True, unified=10, find_renames=True)
                 files_changed = []
                 current_file = None
                 current_patch_lines = []
                 in_header = True
                 for line in raw_diff.splitlines():
                     if line.startswith('diff --git a/'):
                         in_header = False
                         if current_file: files_changed.append({'filename': current_file, 'status': 'A', 'patch': "\n".join(current_patch_lines).strip(), 'old_filename': None})
                         parts = line.split(' b/'); current_file = parts[-1] if len(parts) > 1 else line.split(' a/')[-1]
                         current_patch_lines = [line]
                     elif not in_header and current_file: current_patch_lines.append(line)
                 if current_file: files_changed.append({'filename': current_file, 'status': 'A', 'patch': "\n".join(current_patch_lines).strip(), 'old_filename': None})
                 diff_data['files'] = files_changed
             except Exception as diff_err:
                 diff_data['diff_error'] = f"Could not generate diff for initial commit {commit_sha[:7]}: {diff_err}"
        else:
            # Diff against the first parent for non-initial commits
            parent = commit_obj.parents[0]
            diffs = commit_obj.diff(parent, create_patch=True, R=True) # R=True helps detect renames
            file_diffs = []
            for diff_item in diffs:
                 patch_text = diff_item.diff.decode('utf-8', errors='replace') if diff_item.diff else ""
                 file_diffs.append({
                     'filename': diff_item.b_path if diff_item.b_path else diff_item.a_path, # Use b_path preferentially
                     'old_filename': diff_item.a_path if diff_item.renamed else None,
                     'status': diff_item.change_type, # e.g., 'A', 'D', 'M', 'R', 'T'
                     'patch': patch_text
                 })
            diff_data['files'] = file_diffs

        return diff_data, None # Return dictionary

    except (git.BadName, git.BadObject):
        return None, f"Commit SHA '{commit_sha}' not found in repository."
    except git.GitCommandError as e:
         return None, f"Git command error getting diff for {commit_sha[:7]}: {e.stderr or e}"
    except Exception as e:
        st.error(traceback.format_exc()) # Log details
        return None, f"Unexpected error getting diff for {commit_sha[:7]}: {e}"

# summarize_commit (remains the same)
@st.cache_data(ttl=3600)
def summarize_commit(_commit_data_dict, target_language="English"):
    """Summarizes a commit using the Gemini model and translates if needed."""
    if not _commit_data_dict or not isinstance(_commit_data_dict, dict):
         return f"Could not generate summary: Invalid commit data."

    sha = _commit_data_dict.get('sha', 'N/A')
    prompt = f"""

You are an expert AI assistant specializing in code analysis.
Analyze and summarize the following Git commit in **simple, non-technical language**.
Focus on the **what** and **why** of the change, understandable by someone with minimal programming background. Keep the summary concise (2-3 sentences).
And skip standard leads of noun phrases like "This change..." or "This commit..." etc. leave them out .

Commit SHA: {sha}
Author: {_commit_data_dict.get('author_name', 'N/A')}
Date: {_commit_data_dict.get('authored_datetime', 'N/A')}
Commit Message:
{_commit_data_dict.get('message', 'N/A')}
Please provide the summary in {target_language}."""
    try:
        response = model.generate_content(prompt)
        # Check for valid response content
        if response.parts:
            return response.text.strip()
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             reason = response.prompt_feedback.block_reason.name.replace("_", " ").lower()
             return f"Summary blocked for {sha[:7]} due to {reason} content."
        else:
             return f"Could not generate summary for {sha[:7]} (empty response)."
    except Exception as e:
        return f"Error generating summary for {sha[:7]}: {e}"

# format_commit_context (for page view - remains the same)
def format_commit_context(commits_on_page_dicts, total_filtered_commits):
    """Formats commit data *from the current page* for the page-specific Q&A AI."""
    if not commits_on_page_dicts:
        return f"No commit data displayed on this page.\nTotal commits matching filters: {total_filtered_commits}"
    context_str = (f"Commit data for the current page ({len(commits_on_page_dicts)} commits shown "
                   f"out of {total_filtered_commits} total matching filters):\n\n")
    for commit_dict in commits_on_page_dicts: # Iterate over dictionaries
        context_str += f"Commit SHA: {commit_dict.get('sha', 'N/A')}\nAuthor: {commit_dict.get('author_name', 'N/A')}\nDate: {commit_dict.get('authored_datetime', 'N/A')}\nMessage:\n{commit_dict.get('message', 'N/A')}\n---\n"
    return context_str

# format_chat_history (remains the same)
def format_chat_history(chat_history, max_turns=MAX_HISTORY_TURNS):
    """Formats the chat history list into a string for the AI prompt."""
    if not chat_history: return "No previous conversation history."
    turns_to_include = min(len(chat_history), max_turns * 2)
    relevant_history = chat_history[-turns_to_include:]
    formatted_history = "".join(f"{'User' if msg.get('role') == 'user' else 'Assistant'}: {str(msg.get('content', ''))}\n" for msg in relevant_history)
    if len(chat_history) > turns_to_include: formatted_history = "[...truncated history...]\n" + formatted_history
    return formatted_history.strip()

# answer_question (for page view - remains the same)
def answer_question(question, commit_context, chat_history_list, total_filtered_commits):
    """Uses Gemini to answer a question based on the *current page's* commit context."""
    if not question: return "Please ask a question."
    if not commit_context or "Total commits matching filters: 0" in commit_context:
        return "Cannot answer: No commit data loaded."
    formatted_chat_history = format_chat_history(chat_history_list)
    prompt = f"""You are a helpful AI assistant translating technical Git commit information for a non-technical audience.
Answer the user's latest question based *primarily* on the Provided Commit Data (Current Page). You are aware there are {total_filtered_commits} total commits matching filters, but detailed info is only provided for the current page below. Explain simply. If the answer isn't on this page, say so clearly. Do not guess details from other commits.

Provided Commit Data (Current Page):
{commit_context}
---
Ongoing Conversation History:
{formatted_chat_history}
---
User's Latest Question:
{question}
---
Answer:"""
    try:
        response = model.generate_content(prompt)
        if response.parts: return response.text.strip()
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: return f"Blocked due to {response.prompt_feedback.block_reason.name}."
        else: return "Sorry, received an empty response from the AI."
    except Exception as e: return f"Sorry, an error occurred: {e}"


# ---- RAG Specific Functions (remain the same) ----
def create_commit_text_chunk(commit_dict):
    """Creates a text chunk for embedding from a commit dictionary."""
    return (
        f"Commit: {commit_dict.get('sha', '')}\n"
        f"Author: {commit_dict.get('author_name', '')} <{commit_dict.get('author_email', '')}>\n"
        f"Date: {commit_dict.get('authored_datetime', '')}\n"
        f"Message:\n{commit_dict.get('message', '')}"
    ).strip()

@st.cache_data(show_spinner=False) # Cache the index based on the list of commits
def build_commit_index(_all_commits_list):
    """Builds an embedding index for the provided list of commit dictionaries."""
    if not _all_commits_list:
        return None, None, "No commits to index."

    commit_texts = [create_commit_text_chunk(commit) for commit in _all_commits_list]
    commit_shas = [commit['sha'] for commit in _all_commits_list]

    embeddings_list = []
    errors = []

    # Batching requests to the embedding API for efficiency
    batch_size = 100 # Gemini API batch limit is often 100
    for i in range(0, len(commit_texts), batch_size):
        batch = commit_texts[i:i+batch_size]
        try:
            # Use task_type="RETRIEVAL_DOCUMENT" for indexing content
            result = genai.embed_content(
                model=EMBEDDING_MODEL_NAME,
                content=batch,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings_list.extend(result['embedding'])
        except Exception as e:
            errors.append(f"Error embedding batch starting at index {i}: {e}")
            # Add None placeholders for the failed batch size to keep indices aligned
            embeddings_list.extend([None] * len(batch))

    # Filter out None embeddings due to errors
    valid_embeddings = [emb for emb in embeddings_list if emb is not None]
    valid_indices = [idx for idx, emb in enumerate(embeddings_list) if emb is not None]

    if not valid_embeddings:
         return None, None, f"Failed to generate any embeddings. Errors: {errors}"

    # Store as numpy array
    embeddings_array = np.array(valid_embeddings)
    # Map the row index in the numpy array back to the original commit SHA
    index_to_sha_map = {idx: commit_shas[original_idx] for idx, original_idx in enumerate(valid_indices)}

    error_message = f"Completed indexing with {len(errors)} batch errors." if errors else None
    return embeddings_array, index_to_sha_map, error_message

@st.cache_data(show_spinner=False) # Cache search results for the same query & index
def find_relevant_commits(query: str, embeddings_array: np.ndarray, index_to_sha_map: dict, all_commits_map: dict, top_k: int):
    """Finds the top_k most relevant commits to the query using cosine similarity."""
    if embeddings_array is None or not index_to_sha_map:
        return [], "Index not available."

    try:
        # Embed the query - use RETRIEVAL_QUERY type
        query_embedding_response = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=query,
            task_type="RETRIEVAL_QUERY" # Important: Use query type here
        )
        query_embedding = np.array(query_embedding_response['embedding'])
    except Exception as e:
        return [], f"Failed to embed query: {e}"

    # Calculate cosine similarity
    similarities = [1 - cosine(query_embedding, embeddings_array[i]) for i in range(embeddings_array.shape[0])]
    similarities = np.array(similarities)

    # Get the indices of the top_k highest similarities
    top_k_indices = np.argsort(similarities)[-top_k:][::-1] # Get top k and reverse for descending order

    relevant_shas = [index_to_sha_map[idx] for idx in top_k_indices if idx in index_to_sha_map]
    # Retrieve the full commit dictionaries using the SHAs
    relevant_commits_list = [all_commits_map[sha] for sha in relevant_shas if sha in all_commits_map]

    return relevant_commits_list, None

def answer_global_question(question, relevant_commits, chat_history_list):
    """Uses Gemini to answer a question based on specifically retrieved relevant commits."""
    if not question:
        return "Please ask a question."
    if not relevant_commits:
        # Modify message slightly for clarity
        return "I couldn't find any specific commits that seem highly relevant to your question in the filtered set based on semantic search. You could try rephrasing or asking about different keywords."

    formatted_chat_history = format_chat_history(chat_history_list, max_turns=GLOBAL_MAX_HISTORY_TURNS)

    # Build context string from *only* the relevant commits
    context_str = "Based on my analysis of the repository, here is the most relevant commit information found related to your question:\n\n"
    for commit_dict in relevant_commits:
        context_str += f"Commit SHA: {commit_dict.get('sha', 'N/A')}\n"
        context_str += f"Author: {commit_dict.get('author_name', 'N/A')}\n"
        context_str += f"Date: {commit_dict.get('authored_datetime', 'N/A')}\n"
        context_str += f"Message:\n{commit_dict.get('message', 'N/A')}\n---\n"

    prompt = f"""You are an AI assistant analyzing a Git repository. The user asked a question, and based on semantic search, the following commit(s) were identified as potentially relevant.
Answer the user's **latest question** based *only* on the **Relevant Commit Data Provided** below. If the provided data isn't sufficient, state that clearly. Do not invent information outside this data.

Ongoing Conversation History:
{formatted_chat_history}
---
Relevant Commit Data Provided:
{context_str}
---
User's Latest Question:
{question}
---
Answer:"""

    try:
        response = model.generate_content(prompt)
        if response.parts:
            return response.text.strip()
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             reason = response.prompt_feedback.block_reason.name.replace("_", " ").lower()
             return f"Sorry, couldn't generate answer. Blocked due to {reason} content."
        else:
             return "Sorry, received an empty response from the AI."
    except Exception as e:
        return f"Sorry, an error occurred generating the answer: {e}"


# ---- Session State Initialization (remains the same) ----
default_filter_values = {
    'process_request': False, 'current_page_number': 1, 'total_commit_count': 0,
    'total_pages_local': 1, 'all_commits_list': [], 'chat_history': [], 'repo_path': "",
    'valid_repo_path': None, 'active_sha_filter': None, 'active_start_date': None,
    'active_end_date': None, 'active_author_filter': None, 'active_committer_filter': None,
    'active_message_filter': None, 'active_path_filter': None, 'active_include_merges': False,
    # RAG specific state
    'commit_index_embeddings': None, # Holds the numpy array
    'commit_index_map': None,        # Holds the index -> SHA mapping
    'all_commits_map_for_lookup': {}, # SHA -> commit_dict map for fast retrieval
    'index_error_msg': None,
    'global_chat_history': [],       # Separate history for global Q&A
}
for key, default_value in default_filter_values.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ---- UI ----
st.set_page_config(layout="wide", page_title="AI Local Commit Analyzer")
st.title("🚀 AI-Powered Local Git Commit Analyzer")
st.markdown("Explore, summarize, and ask questions about commits from a **local Git repository**.")

# --- Input Area (remains the same) ---
repo_col, lang_col = st.columns([3, 1])
with repo_col:
    repo_path_input_val = st.text_input(
        "Enter the Path to your Local Git Repository:", key="repo_path_input_widget",
        value=st.session_state.repo_path, help="Full path to the directory containing the .git folder/file."
    )
    if repo_path_input_val != st.session_state.repo_path:
        st.session_state.repo_path = repo_path_input_val
        # Reset everything related to repo data and index on path change
        st.session_state.valid_repo_path = None
        st.session_state.process_request = False
        st.session_state.all_commits_list = []
        st.session_state.chat_history = []
        st.session_state.global_chat_history = []
        st.session_state.current_page_number = 1
        st.session_state.commit_index_embeddings = None
        st.session_state.commit_index_map = None
        st.session_state.all_commits_map_for_lookup = {}
        st.session_state.index_error_msg = None
        st.cache_data.clear() # Clear all Streamlit cache
        st.rerun()

    valid_path = False
    if st.session_state.repo_path:
        is_valid, msg_or_path = validate_repo_path(st.session_state.repo_path)
        if is_valid:
            if msg_or_path != st.session_state.get('valid_repo_path'):
                st.session_state.valid_repo_path = msg_or_path
                st.success(f"Valid repository found: {msg_or_path}", icon="✅")
            valid_path = True
        else:
            if st.session_state.get('valid_repo_path'):
                 st.session_state.valid_repo_path = None; st.session_state.process_request = False; st.session_state.all_commits_list = []
            st.warning(f"Invalid Path: {msg_or_path}", icon="⚠️")
            valid_path = False

with lang_col:
    selected_language = st.selectbox(
        "Select Summary Language:", options=list(SUPPORTED_LANGUAGES.keys()), index=0, key="language_select",
        help="Choose the language for the AI-generated commit summaries."
    )

# --- Filtering Area (remains the same) ---
st.subheader("🔎 Filter Commits (Optional)")
filter_row1_col1, filter_row1_col2, filter_row1_col3 = st.columns(3)
with filter_row1_col1: sha_input = st.text_input("Branch/Tag/SHA:", placeholder="e.g., main, v1.0, a1b2c3d", key="sha_filter", value=st.session_state.active_sha_filter or "").strip()
with filter_row1_col2: start_date_input = st.date_input("From Date:", value=st.session_state.active_start_date, key="start_date_filter", help="Show commits from this date (inclusive).")
with filter_row1_col3: end_date_input = st.date_input("To Date:", value=st.session_state.active_end_date, key="end_date_filter", min_value=start_date_input if start_date_input else None, help="Show commits up to this date (inclusive).")
filter_row2_col1, filter_row2_col2, filter_row2_col3 = st.columns(3)
with filter_row2_col1: author_input = st.text_input("Author:", placeholder="Name or email", key="author_filter", value=st.session_state.active_author_filter or "", help="Filter by author name or email.").strip()
with filter_row2_col2: committer_input = st.text_input("Committer:", placeholder="Name or email", key="committer_filter", value=st.session_state.active_committer_filter or "", help="Filter by committer name or email.").strip()
with filter_row2_col3: path_input = st.text_input("File/Directory Path:", placeholder="e.g., src/utils.py", key="path_filter", value=st.session_state.active_path_filter or "", help="Show commits affecting this specific path.").strip()
filter_row3_col1, filter_row3_col2 = st.columns([3, 1])
with filter_row3_col1: message_input = st.text_input("Commit Message Contains:", placeholder="e.g., fix bug #123", key="message_filter", value=st.session_state.active_message_filter or "", help="Filter by text within commit message (case-insensitive).").strip()
with filter_row3_col2: include_merges_input = st.checkbox("Include Merge Commits", key="include_merges_filter", value=st.session_state.active_include_merges, help="Check to include merge commits.")
is_date_range_invalid = bool(start_date_input and end_date_input and start_date_input > end_date_input)
if is_date_range_invalid: st.warning("Warning: Start date is after end date.", icon="⚠️")

# --- Process Button (remains the same - clears index state) ---
button_disabled_state = not valid_path or is_date_range_invalid
process_button = st.button("Get Commits & Apply Filters", key="process_button", type="primary", disabled=button_disabled_state)
if process_button:
    if st.session_state.valid_repo_path:
        filters_changed = (
            (sha_input if sha_input else None) != st.session_state.active_sha_filter or start_date_input != st.session_state.active_start_date or
            end_date_input != st.session_state.active_end_date or (author_input if author_input else None) != st.session_state.active_author_filter or
            (committer_input if committer_input else None) != st.session_state.active_committer_filter or (message_input if message_input else None) != st.session_state.active_message_filter or
            (path_input if path_input else None) != st.session_state.active_path_filter or include_merges_input != st.session_state.active_include_merges )

        st.session_state.active_sha_filter = sha_input if sha_input else None
        st.session_state.active_start_date = start_date_input; st.session_state.active_end_date = end_date_input
        st.session_state.active_author_filter = author_input if author_input else None
        st.session_state.active_committer_filter = committer_input if committer_input else None
        st.session_state.active_message_filter = message_input if message_input else None
        st.session_state.active_path_filter = path_input if path_input else None
        st.session_state.active_include_merges = include_merges_input

        if filters_changed or not st.session_state.process_request:
             # Reset state including index
             st.session_state.process_request = True; st.session_state.current_page_number = 1; st.session_state.all_commits_list = []
             st.session_state.total_commit_count = 0; st.session_state.total_pages_local = 1; st.session_state.chat_history = []; st.session_state.global_chat_history = []
             st.session_state.commit_index_embeddings = None; st.session_state.commit_index_map = None; st.session_state.all_commits_map_for_lookup = {}
             st.session_state.index_error_msg = None; st.info("Filters applied. Fetching commits..."); st.rerun()
        else: st.info("Filters already applied.")


# --- Display Section (Pagination, Summaries, Page Q&A) ---
if st.session_state.process_request and st.session_state.valid_repo_path:
    st.markdown("---")
    st.subheader("Browse Commits (Paginated)")

    # Fetch commits ONLY if needed (remains same)
    if not st.session_state.all_commits_list and st.session_state.total_commit_count == 0:
        with st.spinner("Reading commits from local repository..."):
            all_commits_data, error = get_local_commits(
                st.session_state.valid_repo_path, st.session_state.active_sha_filter,
                st.session_state.active_start_date, st.session_state.active_end_date,
                st.session_state.active_author_filter, st.session_state.active_committer_filter,
                st.session_state.active_message_filter, st.session_state.active_path_filter,
                st.session_state.active_include_merges )

        if error:
            st.error(f"Failed to read commits: {error}")
            st.session_state.process_request = False; st.session_state.all_commits_list = []; st.session_state.total_commit_count = 0
        elif all_commits_data is not None:
            st.session_state.all_commits_list = all_commits_data
            st.session_state.total_commit_count = len(all_commits_data)
            st.session_state.total_pages_local = max(1, math.ceil(st.session_state.total_commit_count / COMMITS_PER_PAGE))
            # Build the SHA -> commit_dict lookup map needed for RAG retrieval
            st.session_state.all_commits_map_for_lookup = {commit['sha']: commit for commit in all_commits_data}
            if not error: st.success(f"Found {st.session_state.total_commit_count} matching commits.")
        else:
             st.error("Commit data is None."); st.session_state.process_request = False

    # --- PAGINATION & DISPLAY LOGIC ---
    if st.session_state.process_request and st.session_state.all_commits_list is not None:
        total_commits = st.session_state.total_commit_count
        total_pages = st.session_state.total_pages_local
        current_page = st.session_state.current_page_number
        # Page number validation
        current_page = max(1, min(current_page, total_pages))
        st.session_state.current_page_number = current_page

        start_index = (current_page - 1) * COMMITS_PER_PAGE
        end_index = start_index + COMMITS_PER_PAGE
        commits_on_current_page_dicts = st.session_state.all_commits_list[start_index:end_index]

        # Pagination controls (remains same)
        if total_pages > 1:
             pcol1, pcol2 = st.columns([1, 3])
             with pcol1:
                 new_page_num = st.number_input(f"Page (1-{total_pages})", min_value=1, max_value=total_pages, value=current_page, step=1, key=f"page_selector_{total_pages}")
                 if new_page_num != current_page:
                     st.session_state.current_page_number = new_page_num; st.session_state.chat_history = []; st.toast("Page changed, page chat cleared.", icon="ℹ️"); st.rerun()
             with pcol2: # Display commit range
                 if commits_on_current_page_dicts: st.caption(f"Showing commits {start_index + 1} - {start_index + len(commits_on_current_page_dicts)} of {total_commits} total")
                 elif total_commits > 0: st.caption(f"Page {current_page}/{total_pages} (No commits on this page)")
        elif total_commits > 0: st.subheader(f"Commits ({total_commits} total)")

        # --- DETAILED COMMIT VIEW (Restored) ---
        if not commits_on_current_page_dicts and total_commits == 0: st.info("No commits found matching your criteria.")
        elif not commits_on_current_page_dicts and current_page <= total_pages: st.info(f"No commits found on page {current_page}.")
        elif commits_on_current_page_dicts:
            with st.expander(f"View Commit Summaries & Details (Page {current_page})", expanded=False): # Default closed
                st.markdown(f"**Summaries below are in {selected_language}:**")
                progress_bar = st.progress(0, text="Preparing summaries...")
                summaries_container = st.container(height=600) # Scrollable container
                total_on_page = len(commits_on_current_page_dicts)

                for i, commit_dict in enumerate(commits_on_current_page_dicts):
                    # --- Start of restored detail section ---
                    sha = commit_dict.get('sha', 'N/A')
                    message = commit_dict.get('message', 'N/A')
                    author_name = commit_dict.get('author_name', 'N/A')
                    author_email = commit_dict.get('author_email', 'N/A')
                    committer_name = commit_dict.get('committer_name', 'N/A')
                    date_str_iso = commit_dict.get('authored_datetime', 'N/A')
                    try: date_obj = datetime.fromisoformat(date_str_iso); date_display = date_obj.strftime('%Y-%m-%d %H:%M:%S %Z')
                    except: date_display = date_str_iso # Fallback

                    summary = summarize_commit(commit_dict, selected_language)
                    first_line = message.splitlines()[0] if message else "(No commit message)"
                    commit_header = f"[{sha[:7]}] {first_line[:80]}{'...' if len(first_line)>80 else ''}"

                    with summaries_container:
                        st.markdown(f"#### {commit_header}")
                        st.markdown(f"**Summary ({selected_language}):**"); st.markdown(summary);

                        col_details1, col_details2 = st.columns(2)
                        with col_details1:
                            st.caption(f"**Author:** {author_name} ({author_email})")
                            st.caption(f"**SHA:** {sha}")
                        with col_details2:
                            st.caption(f"**Date:** {date_display}")
                            st.caption(f"**Committer:** {committer_name}")

                        show_full_message = st.toggle("Show Full Commit Message", key=f"msg_toggle_{sha}", value=False)
                        if show_full_message: st.code(message, language='text')

                        # Diff Display Logic
                        show_diff = st.toggle("Show Code Changes", key=f"diff_toggle_{sha}", value=False)
                        if show_diff:
                            diff_placeholder = st.empty(); diff_placeholder.markdown("_Fetching code changes..._")
                            diff_data, diff_error = get_commit_diff(st.session_state.valid_repo_path, sha)
                            if diff_error: diff_placeholder.error(f"Could not load diff: {diff_error}")
                            elif diff_data:
                                diff_placeholder.empty()
                                files = diff_data.get('files', [])
                                diff_gen_error = diff_data.get('diff_error')
                                if diff_gen_error: st.warning(f"Note generating diff: {diff_gen_error}")
                                if files:
                                    st.markdown("**Code Changes:**")
                                    diff_found_display = False
                                    status_map = {'A': 'Added', 'D': 'Deleted', 'M': 'Modified', 'R': 'Renamed', 'T': 'TypeChanged', 'C':'Copied', 'U':'Unmerged', '?':'Unknown'}
                                    for file in files:
                                        filename = file.get('filename', 'Unknown'); patch = file.get('patch')
                                        status_char = file.get('status', '?'); status_desc = status_map.get(status_char, status_char)
                                        display_filename = f"{file['old_filename']} -> {filename}" if file.get('old_filename') else filename
                                        file_info_col, file_diff_col = st.columns([1, 4])
                                        with file_info_col: st.caption(f"`{display_filename}`"); st.caption(f"Status: {status_desc}")
                                        with file_diff_col:
                                            if patch: st.code(patch, language='diff', line_numbers=True); diff_found_display = True
                                            else: st.caption("_No textual diff content._")
                                        st.divider() # Divider per file
                                    if not diff_found_display and not diff_gen_error: st.info("No textual diffs found in this commit.")
                                elif not diff_gen_error: st.info("No files reported changed in this commit's diff.")
                            else: diff_placeholder.warning("Could not retrieve valid diff details.")
                        st.divider() # Commit separator
                    # --- End of restored detail section ---
                    progress_bar.progress((i + 1) / total_on_page, text=f"Processing summaries... {i+1}/{total_on_page}")
                progress_bar.empty()

            # --- Page-Specific Chatbot Q&A (with corrected syntax) ---
            with st.expander(f"❓ Ask Questions About Commits on Page {current_page}", expanded=False):
                 st.caption(f"AI answers based on the **{len(commits_on_current_page_dicts)} commits on page {current_page}** only.")
                 chat_container_page = st.container(height=300)
                 with chat_container_page:
                     # Display existing history
                     for message in st.session_state.chat_history:
                         with st.chat_message(message["role"]): st.markdown(message["content"])

                 # Get user input
                 user_question_page = st.chat_input("Ask about commits on this page...", key="qa_input_page")

                 # Process if user entered a question
                 if user_question_page:
                     # Append user message to state
                     st.session_state.chat_history.append({"role": "user", "content": user_question_page})

                     # --- **** CORRECTED SYNTAX HERE **** ---
                     # Display user message immediately
                     with st.chat_message("user"):
                         st.markdown(user_question_page)
                     # --- **** END CORRECTION **** ---

                     # Get AI answer
                     with st.spinner("Thinking..."):
                         commit_context = format_commit_context(commits_on_current_page_dicts, total_commits)
                         ai_answer = answer_question(user_question_page, commit_context, st.session_state.chat_history, total_commits)
                         st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})

                     # Rerun to display the AI answer in the chat container
                     st.rerun()

    # ---- GLOBAL RAG Q&A SECTION (remains the same) ----
    st.markdown("---")
    st.subheader(f"💬 Ask About All {st.session_state.total_commit_count} Filtered Commits")
    st.caption("Uses AI search (RAG) to find relevant commits across all pages based on your query.")

    # Display index status/errors
    if st.session_state.index_error_msg:
        st.warning(f"Indexing issue: {st.session_state.index_error_msg}")

    # Only allow global chat if commits exist
    global_chat_disabled = st.session_state.total_commit_count == 0
    if global_chat_disabled:
        st.info("Load commits using the filters above to enable global Q&A.")

    chat_container_global = st.container(height=400)
    with chat_container_global:
        # Display global chat history
        for message in st.session_state.global_chat_history:
             with st.chat_message(message["role"]): st.markdown(message["content"])

    # Global chat input
    user_question_global = st.chat_input(
        "Ask about any commit in the filtered set...",
        key="qa_input_global",
        disabled=global_chat_disabled
    )

    if user_question_global and not global_chat_disabled:
        # Append and display user message immediately
        st.session_state.global_chat_history.append({"role": "user", "content": user_question_global})
        with chat_container_global:
            with st.chat_message("user"): st.markdown(user_question_global)

        ai_response_global = ""
        with st.spinner("Analyzing all commits... (Building index if needed)"):
            # --- RAG Workflow ---
            # 1. Build index if it doesn't exist for the current commit list
            if st.session_state.commit_index_embeddings is None and st.session_state.all_commits_list:
                 st.write("_Building semantic index for commits..._") # Temporary message
                 embeddings_array, index_map, index_err = build_commit_index(st.session_state.all_commits_list)
                 st.session_state.commit_index_embeddings = embeddings_array
                 st.session_state.commit_index_map = index_map
                 st.session_state.index_error_msg = index_err
                 if index_err: st.warning(f"Indexing issue: {index_err}")
                 # No rerun here, proceed directly to search

            # 2. Find relevant commits (if index exists)
            relevant_commits_list = []
            search_error = None
            if st.session_state.commit_index_embeddings is not None:
                relevant_commits_list, search_error = find_relevant_commits(
                    user_question_global,
                    st.session_state.commit_index_embeddings,
                    st.session_state.commit_index_map,
                    st.session_state.all_commits_map_for_lookup, # Pass the SHA->commit map
                    top_k=NUM_RELEVANT_COMMITS
                )
                if search_error:
                    st.error(f"Search error: {search_error}") # Show search error

            # 3. Generate answer using relevant commits
            if search_error:
                 ai_response_global = f"I encountered an error trying to search for relevant commits: {search_error}"
            elif not relevant_commits_list and st.session_state.commit_index_embeddings is not None:
                 # Handle case where search ran but found nothing - call answer func with empty list
                 ai_response_global = answer_global_question(user_question_global, [], st.session_state.global_chat_history)
            elif relevant_commits_list:
                 # We have relevant commits, generate the answer
                 ai_response_global = answer_global_question(
                     user_question_global,
                     relevant_commits_list,
                     st.session_state.global_chat_history
                 )
            elif st.session_state.commit_index_embeddings is None:
                 ai_response_global = "Could not answer because the commit index is not available (likely due to an earlier error)."
            else: # Should not happen easily
                 ai_response_global = "An unexpected state occurred during the RAG process."

        # 4. Append AI response to history
        st.session_state.global_chat_history.append({"role": "assistant", "content": ai_response_global})
        # Rerun to display the AI message in the global chat container
        st.rerun()


# --- Footer/Initial State (remains same) ---
elif not st.session_state.process_request:
     if not st.session_state.repo_path: st.info("Enter the path to a local Git repository above and click 'Get Commits' to begin.")
     elif not st.session_state.valid_repo_path: pass # Validation warning is shown above when input changes

st.divider()
# Display current date using the variable defined earlier if needed, otherwise use datetime.now()
current_date_display = datetime.now(timezone.utc).date().isoformat()
st.caption(f"Date: {current_date_display} (UTC)")