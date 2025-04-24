import subprocess

def get_git_log_with_diff(repo_path: str, max_commits: int = 5) -> str:
    try:
        result = subprocess.run(
           # ["git", "log", f"-n{max_commits}", "-p", "--no-color"],
            ["git", "log", f"-n{max_commits}", "-p"],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running git log:", e.stderr)
        return ""

# Example usage
repo_dir = "../../../commit-messages-guide"
diff_output = get_git_log_with_diff(repo_dir, max_commits=1)
# parser should be here
diff_output_list = diff_output.split('commit')
print(diff_output_list)