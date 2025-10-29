#!/bin/bash
set -euo pipefail

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing arguments"
    echo "Usage: $0 <python_file> --gpu=<id>"
    exit 1
fi

PYTHON_FILE=$1
GPU_ARG=$2

# Parse GPU_ID
GPU_ID="${GPU_ARG#*=}"

SESSION_NAME="$(id -un)"
CMD="CUDA_VISIBLE_DEVICES=$GPU_ID pixi run python $PYTHON_FILE"

# tmux new-session -d -s $SESSION_NAME -A
if ! tmux has -t="$SESSION_NAME" 2>/dev/null; then
    tmux new -s "$SESSION_NAME" -d
fi
tmux new-window -t "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" "$CMD" C-m

echo "Running command '$CMD' in tmux session '$SESSION_NAME' on GPU $GPU_ID"
