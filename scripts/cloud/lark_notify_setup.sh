#!/usr/bin/env bash
# Cloud agent install step: provision lark-cli + the Lark/Feishu notification
# skills so a cloud-agent review task can DM a summary to the maintainer.
#
# Safe to run on every VM startup:
#   - Exits 0 without doing anything when LARK_APP_ID / LARK_APP_SECRET are
#     absent, so cloud agents with no Lark credentials still start cleanly.
#   - Idempotent: skips lark-cli / skill installs that are already present.
#   - Uses a user-writable npm global prefix (~/.npm-global) so `npm i -g`
#     does not hit EACCES on the shared base image.
set -euo pipefail

if [[ -z "${LARK_APP_ID:-}" || -z "${LARK_APP_SECRET:-}" ]]; then
  echo "lark: LARK_APP_ID/LARK_APP_SECRET not set, skipping lark-cli setup"
  exit 0
fi

# Route npm global installs to a writable location and make the resulting
# binaries discoverable for the rest of this script.
export NPM_CONFIG_PREFIX="${NPM_CONFIG_PREFIX:-$HOME/.npm-global}"
mkdir -p "$NPM_CONFIG_PREFIX/bin"
export PATH="$NPM_CONFIG_PREFIX/bin:$PATH"

if ! command -v lark-cli >/dev/null 2>&1; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "lark: npm not found; install Node.js >=16 in the base image first" >&2
    exit 1
  fi
  npm install -g @larksuite/cli
fi

# Skill files (SKILL.md + references/) install fine globally; the optional
# PromptScript component is not supported for -g installs, which is harmless.
for skill in lark-shared lark-im; do
  if [[ ! -d "$HOME/.agents/skills/$skill" ]]; then
    npx -y skills add larksuite/cli -s "$skill" -y -g || true
  fi
done

# Configure the app from the injected secrets (non-interactive) and default to
# bot identity so notifications can be sent without an OAuth browser flow.
printf '%s' "$LARK_APP_SECRET" |
  lark-cli config init \
    --app-id "$LARK_APP_ID" \
    --app-secret-stdin \
    --brand "${LARK_BRAND:-feishu}"

lark-cli config default-as bot
lark-cli doctor --offline
