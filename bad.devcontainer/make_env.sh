#! /usr/bin/env bash

# Script to create a file called .devcontainer/.env, with the following
# UID=<userid>
# GID=<groupid>
# UNAME=<username>

USER_ID=$(id -u)
GROUP_ID=$(id -g)
USER_NAME=$(id -un)

cat <<EOF > .devcontainer/.env
UID=${USER_ID}
GID=${GROUP_ID}
UNAME=${USER_NAME}
EOF

