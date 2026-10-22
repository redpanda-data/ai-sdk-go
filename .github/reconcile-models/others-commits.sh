#!/usr/bin/env bash
# Prints, comma-separated, the short SHA of each bot PR commit that the bot
# didn't both author and commit: a fix someone pushed, an "Update branch"
# merge, a bot commit someone amended, or one GitHub can't tie to an account.
# Reads GitHub's list of the PR's commits (pulls/N/commits) on stdin.
# Args: the bot's GitHub login.
set -euo pipefail
jq -r --arg bot "$1" '[.[] | select(.author.login != $bot or .committer.login != $bot) | .sha[0:7]] | join(", ")'
