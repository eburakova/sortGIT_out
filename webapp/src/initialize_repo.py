import subprocess

def get_git_log_with_diff():
    result = subprocess.run(
        ['git', 'log', '-p', '--no-color'],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout

log_output = get_git_log_with_diff()

# Write the raw output
with open('repo.log', 'w') as f:
    f.write(log_output)

# Parse and create a database


# Summarize the differences and commit messages


