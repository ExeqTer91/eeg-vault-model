import os
import base64
import requests

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
REPO = "ExeqTer91/eeg-phi-golden-ratio"
BRANCH = "main"

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_sha(path):
    url = f"https://api.github.com/repos/{REPO}/contents/{path}?ref={BRANCH}"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get('sha')
    return None

def upload_file(local_path, repo_path):
    with open(local_path, 'rb') as f:
        content = base64.b64encode(f.read()).decode()
    
    url = f"https://api.github.com/repos/{REPO}/contents/{repo_path}"
    sha = get_sha(repo_path)
    
    data = {
        "message": f"Add {repo_path}",
        "content": content,
        "branch": BRANCH
    }
    if sha:
        data["sha"] = sha
        data["message"] = f"Update {repo_path}"
    
    r = requests.put(url, headers=headers, json=data)
    if r.status_code in [200, 201]:
        print(f"✓ {repo_path}")
        return True
    else:
        print(f"✗ {repo_path}: {r.status_code}")
        return False

# Find all Python scripts
files = []
for f in os.listdir('.'):
    if f.endswith('.py') and f != 'upload_to_github.py' and f != 'upload_scripts.py':
        files.append((f, f))

# Add eeg-processing scripts
if os.path.exists('eeg-processing'):
    for f in os.listdir('eeg-processing'):
        if f.endswith('.py'):
            files.append((f"eeg-processing/{f}", f"eeg-processing/{f}"))

# Add any other important files
for f in ['requirements.txt', '.streamlit/config.toml']:
    if os.path.exists(f):
        files.append((f, f))

print(f"Uploading {len(files)} scripts...")
success = 0
for local, remote in files:
    if os.path.exists(local):
        if upload_file(local, remote):
            success += 1

print(f"\nDone: {success}/{len(files)} files uploaded")
