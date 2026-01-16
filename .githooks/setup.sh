#!/bin/bash
# Setup script for git hooks
# Run this once after cloning the repository

set -e

# Configure git to use .githooks directory
git config core.hooksPath .githooks

echo "Git hooks configured."
echo "The pre-commit hook will run golangci-lint before each commit."
echo "To disable: git config --unset core.hooksPath"
