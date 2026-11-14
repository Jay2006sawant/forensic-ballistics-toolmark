#!/usr/bin/env bash
# Deploy to Hugging Face Spaces (run after: hf auth login)
set -euo pipefail
cd "$(dirname "$0")"

SPACE_NAME="${1:-forensic-ballistics-toolmark}"
HF_USER="${2:-aryan}"
LOGGED_IN="$(hf auth whoami 2>/dev/null | sed -n 's/^user:[[:space:]]*//p' | head -1)"
if [[ -z "${LOGGED_IN}" ]]; then
  LOGGED_IN="$(hf auth whoami 2>/dev/null | grep -v '^user:' | head -1 | tr -d '[:space:]')"
fi
if [[ -z "${LOGGED_IN}" ]]; then
  echo "Not logged in. Run: hf auth login"
  exit 1
fi
if [[ "${LOGGED_IN}" != "${HF_USER}" ]]; then
  echo "Logged in as ${LOGGED_IN}, but Space owner is set to ${HF_USER}."
  echo "Log in as ${HF_USER}: hf auth login   OR pass his username only when logged in as him."
  exit 1
fi

REPO_ID="${HF_USER}/${SPACE_NAME}"

if [[ ! -d .git ]]; then
  git init -b main
fi

git add app.py compare_core.py report_builder.py styles.css requirements.txt README.md .gitignore deploy_hf.sh
git add test_compare.py 2>/dev/null || true
if ! git diff --cached --quiet; then
  git commit -m "Update forensic ballistics examination app"
fi

hf repo create "$REPO_ID" --repo-type space --space_sdk gradio --exist-ok

if ! git remote get-url origin &>/dev/null; then
  git remote add origin "https://huggingface.co/spaces/${REPO_ID}"
else
  git remote set-url origin "https://huggingface.co/spaces/${REPO_ID}"
fi

git push -u origin main --force

echo ""
echo "Space URL: https://huggingface.co/spaces/${REPO_ID}"