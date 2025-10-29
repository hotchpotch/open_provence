#!/usr/bin/env bash
set -euo pipefail

export GIT_PAGER=cat

release_root="output/release_models"

if [[ ! -d "${release_root}" ]]; then
  echo "Missing ${release_root}; nothing to do."
  exit 0
fi

repos=()
while IFS= read -r -d '' gitdir; do
  repos+=("$(dirname "${gitdir}")")
done < <(find "${release_root}" -mindepth 1 -maxdepth 3 -type d -name ".git" -print0)

if [[ ${#repos[@]} -eq 0 ]]; then
  echo "No Git repositories found under ${release_root}."
  exit 0
fi

mapfile -t repos < <(printf '%s\n' "${repos[@]}" | sort -u)

for repo_dir in "${repos[@]}"; do
  echo "=== ${repo_dir} ==="
  (
    cd "${repo_dir}"
    git status --short --untracked-files=all
  )
done

read -r -p "Execute sync now? y/N " confirm
if [[ ! ${confirm} =~ ^[Yy]$ ]]; then
  echo "Aborted." >&2
  exit 1
fi

for repo_dir in "${repos[@]}"; do
  echo "Syncing ${repo_dir}"
  ( 
    cd "${repo_dir}"
    git add -A
    git commit --allow-empty -m "update"
    git push
  )
done
