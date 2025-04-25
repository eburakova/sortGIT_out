
def define_processing_mode(commit_data, max_commits=500, max_diff_string_length=500):
    if len(commit_data) > max_commits:
        return 'Prefilter' # Prefiltering mandatory!
    diff_lengths = [len(file['diff']) for commit in commit_data for file in commit['files_changed']]
    print(diff_lengths)
    if max(diff_lengths) > max_diff_string_length:
        return 'Summaries' # Reading only commit summaries
    else:
        return 'Full' # Reading code and commit messages