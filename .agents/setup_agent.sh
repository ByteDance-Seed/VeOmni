#!/usr/bin/env bash
# Sets up a .<agent_name>/ directory with symlinks to shared .agents/ resources.
# Usage: bash .agents/setup_agent.sh <agent_name>
# Example: bash .agents/setup_agent.sh gemini
set -euo pipefail

AGENT_NAME="${1:?Usage: bash .agents/setup_agent.sh <agent_name> (e.g., gemini, codex)}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/.${AGENT_NAME}"

if [[ -d "$TARGET_DIR" ]]; then
    echo "Directory .${AGENT_NAME}/ already exists — skipping."
    exit 0
fi

mkdir -p "$TARGET_DIR"

# Symlink shared resources
ln -s ../.agents/skills    "$TARGET_DIR/skills"
ln -s ../.agents/knowledge "$TARGET_DIR/knowledge"
ln -s ../.agents/README.md "$TARGET_DIR/README.md"

# Exclude from git (local only, not committed)
EXCLUDE_FILE="${REPO_ROOT}/.git/info/exclude"
EXCLUDE_ENTRY=".${AGENT_NAME}/"
if ! grep -qF "$EXCLUDE_ENTRY" "$EXCLUDE_FILE" 2>/dev/null; then
    echo "$EXCLUDE_ENTRY" >> "$EXCLUDE_FILE"
    echo "Added .${AGENT_NAME}/ to .git/info/exclude"
fi

echo "Created .${AGENT_NAME}/ with symlinks to .agents/ resources."
