import subprocess

def get_git_log_with_diff(PATH_TO_ROOT):
    result = subprocess.run(
        ['git', 'log', '-p', '--no-color'],
        cwd=PATH_TO_ROOT,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout