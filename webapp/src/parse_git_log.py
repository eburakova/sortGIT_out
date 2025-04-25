import re

def parse_git_log_with_diff(log_text: str):
    commits = []
    current_commit = None
    current_diff = []
    current_file = None
    collecting_diff = False

    lines = log_text.splitlines()
    for line in lines:
        if line.startswith("commit "):
            if current_commit:
                if current_file and current_diff:
                    current_commit["files_changed"].append({
                        "file": current_file,
                        "diff": "\n".join(current_diff)
                    })
                commits.append(current_commit)
            current_commit = {
                "commit": line.split()[1],
                "author": "",
                "date": "",
                "message": "",
                "files_changed": []
            }
            current_diff = []
            current_file = None
            collecting_diff = False

        elif line.startswith("Author:"):
            current_commit["author"] = line[len("Author:"):].strip()

        elif line.startswith("Date:"):
            current_commit["date"] = line[len("Date:"):].strip()

        elif line.startswith("diff --git"):
            if current_file and current_diff:
                current_commit["files_changed"].append({
                    "file": current_file,
                    "diff": "\n".join(current_diff)
                })
            match = re.search(r' a/(\S+)', line)
            current_file = match.group(1) if match else "unknown"
            current_diff = [line]
            collecting_diff = True

        elif collecting_diff:
            current_diff.append(line)

        elif line.strip() and current_commit["message"] == "":
            current_commit["message"] = line.strip()

    # Add the final commit if it exists
    if current_commit:
        if current_file and current_diff:
            current_commit["files_changed"].append({
                "file": current_file,
                "diff": "\n".join(current_diff)
            })
        commits.append(current_commit)

    return commits