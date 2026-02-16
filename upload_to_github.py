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
    """Get SHA of existing file (needed for updates)"""
    url = f"https://api.github.com/repos/{REPO}/contents/{path}?ref={BRANCH}"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get('sha')
    return None

def upload_file(local_path, repo_path):
    """Upload or update a file on GitHub"""
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
        print(f"✗ {repo_path}: {r.status_code} - {r.text[:100]}")
        return False

# Files to upload
files = [
    # Figures
    ("frontiers_figures.zip", "frontiers_figures.zip"),
    ("supplementary_materials.zip", "supplementary_materials.zip"),
    # Results
    ("eeg-processing/results/physionet_results.csv", "eeg-processing/results/physionet_results.csv"),
    ("eeg-processing/results/physionet_combined_results.csv", "eeg-processing/results/physionet_combined_results.csv"),
    # Core scripts
    ("app.py", "app.py"),
    ("replit.md", "replit.md"),
]

# Add figure files
for f in os.listdir("figures_frontiers"):
    if f.endswith(('.tiff', '.jpg', '.png')):
        files.append((f"figures_frontiers/{f}", f"figures_frontiers/{f}"))

# Add supplementary files
for f in os.listdir("supplementary"):
    files.append((f"supplementary/{f}", f"supplementary/{f}"))

print(f"Uploading {len(files)} files to GitHub...")
success = 0
for local, remote in files:
    if os.path.exists(local):
        if upload_file(local, remote):
            success += 1
    else:
        print(f"⚠ {local} not found")

print(f"\nDone: {success}/{len(files)} files uploaded")
