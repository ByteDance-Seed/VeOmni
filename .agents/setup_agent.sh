#!/usr/bin/env bash
# Sets up a .<agent_name>/ directory with symlinks to shared .agents/ resources.
# Usage: bash .agents/setup_agent.sh <agent_name>
# Example: bash .agents/setup_agent.sh gemini
set -euo pipefail

AGENT_NAME="${1:?Usage: bash .agents/setup_agent.sh <agent_name> (e.g., gemini, codex)}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/.${AGENT_NAME}"

if [[ -d "$TARGET_DIR" ]]; then
    ERRORS=()
    for resource in skills knowledge; do
        link="$TARGET_DIR/$resource"
        expected="../.agents/$resource"
        if [[ ! -e "$link" && ! -L "$link" ]]; then
            ERRORS+=("  - $resource: missing. Expected symlink to $expected")
        elif [[ ! -L "$link" ]]; then
            ERRORS+=("  - $resource: is a regular file/directory, not a symlink. Expected symlink to $expected")
        elif [[ "$(readlink "$link")" != "$expected" ]]; then
            ERRORS+=("  - $resource: symlink points to '$(readlink "$link")' instead of '$expected'")
        elif [[ ! -e "$link" ]]; then
            ERRORS+=("  - $resource: dangling symlink (target $expected does not exist)")
        fi
    done

    if [[ ${#ERRORS[@]} -gt 0 ]]; then
        echo "ERROR: .${AGENT_NAME}/ exists but has broken resource links:" >&2
        printf '%s\n' "${ERRORS[@]}" >&2
        echo "" >&2
        echo "To fix, try one of:" >&2
        echo "  1. Remove the directory and re-run:  rm -rf .${AGENT_NAME} && bash .agents/setup_agent.sh ${AGENT_NAME}" >&2
        echo "  2. Manually recreate the symlinks:" >&2
        echo "       ln -sf ../.agents/skills    .${AGENT_NAME}/skills" >&2
        echo "       ln -sf ../.agents/knowledge .${AGENT_NAME}/knowledge" >&2
        exit 1
    fi

    echo "Directory .${AGENT_NAME}/ already exists and looks healthy — skipping."
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
