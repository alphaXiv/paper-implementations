#!/bin/bash
set -e

# DeepCode Sandbox Entrypoint

# Ensure we are in the workspace directory
mkdir -p /workspace
cd /workspace

# Set PYTHONPATH to include the workspace
export PYTHONPATH="${PYTHONPATH}:/workspace"

# If arguments are passed, execute them
if [ "$#" -gt 0 ]; then
    exec "$@"
else
    # Default behavior: keep container running for MCP exec_run commands
    echo "DeepCode Sandbox Ready. Waiting for commands..."
    # Use tail -f /dev/null to keep the container alive indefinitely
    exec tail -f /dev/null
fi
