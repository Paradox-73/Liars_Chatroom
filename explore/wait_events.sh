#!/bin/bash
# Waits up to $1 seconds (default 170) for a teammate comment on the repo's issues after $2 (UTC, ISO) or for one of
# the paradox73 kernels named in the remaining arguments to leave the RUNNING/QUEUED state; prints what it saw.
# bash explore/wait_events.sh 170 2026-11-13T21:40:00Z bake-final-nocap
limit=${1:-170}; since=$2; shift 2
cd "$(dirname "$0")/.." || exit 1
K=../.venv/Scripts/kaggle.exe
end=$(( $(date +%s) + limit ))
while [ "$(date +%s)" -lt "$end" ]; do
  n=$(gh api "repos/Paradox-73/Liars_Chatroom/issues/comments?since=$since&per_page=50" \
        --jq '[.[] | select(.user.login != "Paradox-73")] | length' 2>/dev/null)
  if [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -gt 0 ]; then
    echo "comments: $n new"
    gh api "repos/Paradox-73/Liars_Chatroom/issues/comments?since=$since&per_page=50" \
      --jq '.[] | select(.user.login != "Paradox-73") | "#\(.issue_url | split("/") | last) \(.user.login): \(.body | .[0:2500])"'
    exit 0
  fi
  for k in "$@"; do
    s=$($K kernels status "paradox73/$k" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*')
    if [ -n "$s" ] && [ "$s" != "KernelWorkerStatus.RUNNING" ] && [ "$s" != "KernelWorkerStatus.QUEUED" ]; then
      echo "kernel $k: $s"
      exit 0
    fi
  done
  sleep 40
done
echo "nothing new by $(date '+%H:%M')"
