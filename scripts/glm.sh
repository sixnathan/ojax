#!/bin/sh
set -eu
key=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.pi/agent/auth.json')))['zai']['key'])")
model="${GLM_MODEL:-glm-5-turbo}"
slots="${GLM_SLOTS:-/tmp/glm-slots}"
mkdir -p "$slots"
find "$slots" -mindepth 1 -maxdepth 1 -type d -mmin +20 -exec rmdir {} \; 2>/dev/null || true
acquired=""
while [ -z "$acquired" ]; do
  for i in 1 2; do
    if mkdir "$slots/$i" 2>/dev/null; then acquired="$slots/$i"; break; fi
  done
  [ -z "$acquired" ] && sleep 3
done
trap 'rmdir "$acquired" 2>/dev/null' EXIT INT TERM
env ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic ANTHROPIC_AUTH_TOKEN="$key" ANTHROPIC_MODEL="$model" claude -p --model "$model" "$@"
