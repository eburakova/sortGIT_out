import google.generativeai as genai
import json
import streamlit as st

GEMINI_MODEL = "gemini-1.5-flash"
model = genai.GenerativeModel(GEMINI_MODEL)

def _summarize_commit(commit_info, target_language="English"):
    """Summarizes a commit using the Gemini model and translates if needed."""
    author_info = commit_info.get('author', {})
    commit_message = commit_info.get('message', 'N/A')
    author = commit_info.get('name', 'N/A')
    date_str = commit_info.get('date', 'N/A')
    sha = commit_info.get('sha', 'N/A')
    prompt = f"""
You are an expert AI assistant specializing in code analysis.
Analyze and summarize the following Git commit in **simple, non-technical language**.
Focus on the **what** and **why** of the change, understandable by someone with minimal programming background. Keep the summary concise (2-3 sentences).

Commit SHA: {sha}
Commit Message: {commit_message}
Author: {author}
Date: {date_str}
"""
    if target_language != "English": prompt += f"\n\nPlease provide the summary exclusively in {target_language}."
    else: prompt += f"\n\nPlease provide the summary in English."
    try:
        response = model.generate_content(prompt)
        if response.parts: return response.text
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             reason = response.prompt_feedback.block_reason.name.replace("_", " ").lower()
             return f"Could not generate summary for {sha[:7]}. Blocked due to {reason} content."
        else: return f"Could not generate summary for {sha[:7]}. Empty/invalid response from AI."
    except Exception as e: return f"Error generating summary for {sha[:7]}: {e}"


def _summarize_commit_message(commit_message, commit_hash='', target_language="English"):
    """Summarizes a commit using the Gemini model and translates if needed."""
    prompt = f"""
You are an expert AI assistant specializing in code analysis.
Analyze and summarize the following Git commit in **simple, non-technical language**.
Focus on the **what** and **why** of the change, understandable by someone with minimal programming background. Keep the summary concise (2-3 sentences).

Commit Message: {commit_message}
"""
    if target_language != "English": prompt += f"\n\nPlease provide the summary exclusively in {target_language}."
    else: prompt += f"\n\nPlease provide the summary in English."
    try:
        response = model.generate_content(prompt)
        if response.parts: return response.text
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             reason = response.prompt_feedback.block_reason.name.replace("_", " ").lower()
             return f"Could not generate summary for {commit_hash}. Blocked due to {reason} content."
        else: return f"Could not generate summary for {commit_hash}. Empty/invalid response from AI."
    except Exception as e: return f"Error generating summary for {commit_hash}: {e}"

def _summarize_difference(difference_string, target_language="English"):
    """Summarizes a difference data."""
    prompt = """
You are an expert AI assistant specializing in code analysis. You will get a file of this format:

diff --git a/<file-path> b/<file-path>

---@@ -<starting line>,<number of lines> +<starting line>,<number of lines> @@
<diff content>


As the summarizer, you will then summarize the changes by explaining what lines were added or removed in the commit.

Output strictly in json format.

**Example**

*Input:*

diff --git a/README.md b/README.md
index e544c8d..85f8e30 100644
--- a/README.md
+++ b/README.md
@@ -25,6 +25,7 @@ It may help you to learn what a commit is, why it is important to write good mes
 - [Française](README_fr-FR.md)
 - [پارسی](README_fa-IR.md)
 - [Polish](README_pl-PL.md)
+- [Azərbaycanca](README_az-AZ.md)

 ## What is a "commit"?

diff --git a/README_az-AZ.MD b/README_az-AZ.MD
new file mode 100644
index 0000000..f372f4e
--- /dev/null
+++ b/README_az-AZ.MD
@@ -0,0 +1,494 @@
+# Commit mesajları təlimatı
+
+[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/RomuloOliveira)
+
+Bu təlimat commit mesajlarının əhəmiyyətini və onların daha yaxşı necə yazılacağını izah edir.
+
+
+### rebase -i
+
+Commitləri squash/sıxışdırmaq, mesajları redaktə etmək, commitləri yenidən yazmaq/silmək/sıralarını düzəltmək vs s. üçün istifadə olunur.
+
+

+pick 002a7cc Improve description and update document title
+pick 897f66d Add contributing section
+pick e9549cf Add a section of Available languages
+pick ec003aa Add "What is a commit" section"
+pick bbe5361 Add source referencing as a point of help wanted
+
+# Rebase 9e6dc75..9b81c72 onto 9e6dc75 (15 commands)
+#
+# Commands:
+# p, pick = use commit
+# r, reword = use commit, but edit the commit message
+# e, edit = use commit, but stop for amending
+# s, squash = use commit, but meld into the previous commit
+# However, if you remove everything, the rebase will be aborted.
+#
+# Note that empty commits are commented out
+

+
+#### fixup
+
+Commitləri asanlıqla və daha mürəkkəb bir rebase-ə ehtiyac olmadan təmizləmək üçün istifadə olunur.
+[Bu yazıda](http://fle.github.io/git-tip-keep-your-branch-clean-with-fixup-and-autosquash.html), necə və nə vaxt ediləcəyinə dair gözəl nümunələr var.
+
+### cherry-pick
+
+Səhv branch-da etdiyiniz commiti yenidən kodlaşdırma etmədən tətbiq etmək lazım olduqda bu çox faydalıdır.
+
+Nümunə:
+
+

+$ git cherry-pick 790ab21
+[master 094d820] Fix English grammar in Contributing
+ Date: Sun Feb 25 23:14:23 2018 -0300
+ 1 file changed, 1 insertion(+), 1 deletion(-)
+
+
+Deyək ki, aşağıdakı kimi bir _diff_'imiz var:

*Output:*

{
  "change_type": "natural_language" | "code" | "mixed",
  "files": [
    {
      "path": "README.md",
      "status": "modified" | "added" | "deleted",
      "language": "markdown" | "plaintext" | "az-AZ" | "javascript" | "python" | etc.,
      "summary": "Added a link to Azerbaijani translation."
    },
    {
      "path": "README_az-AZ.MD",
      "status": "added",
      "language": "az-AZ",
      "summary": "Created a full Azerbaijani translation of the commit message guide."
    }
  ],
  "overall_summary": "Added Azerbaijani translation and updated language links in documentation."
}

Summarize this difference message:

""" + difference_string

    if target_language != "English": prompt += f"\n\nPlease provide the summary exclusively in {target_language}."
    else: prompt += f"\n\nPlease provide the summary in English."
    try:
        response = model.generate_content(prompt)
        if response.parts:
            llm_out = response.text
            json_llm_out = llm_out.replace('```json', '').replace('```', '')
            out_dict = json.loads(json_llm_out)
            return out_dict
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             reason = response.prompt_feedback.block_reason.name.replace("_", " ").lower()
             out_dict = {'change_type': 'mixed',
                     'files': [{'path': '[NONE]',
                       'status': '[NA]',
                       'language': '[NA]',
                       'summary': "[NA]"}],
                     'overall_summary': reason}
             return out_dict
        else:
            return f"Empty/invalid response from AI."
    except Exception as e:
        return f"Error generating summary: {e}"


def summarize_commit_info(commit_data, cache_path):
    for j, commit in enumerate(commit_data):
        commit['summary'] = _summarize_commit_message(commit['message'])
        for i, _ in enumerate(commit['files_changed']):
            try:
                diff_message = commit['files_changed'][i]['diff']
                commit['files_changed'][i]['diff_summary'] = _summarize_difference(diff_message)
            except Exception as e:
                st.warning(f"No summary available for {commit['commit']}")
                commit['files_changed'][i]['diff_summary'] = "No summary available"
        with open(cache_path, 'w') as f:
            json.dump(commit_data, f, indent=2)
        st.markdown(f"```Commit processed: {commit['commit']} \t {commit['message']}```")
    return commit_data
