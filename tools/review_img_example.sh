#!/usr/bin/env bash
# review_img_example.sh - Example invocation of review_img.sh using the provided user profile image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CLI="${SCRIPT_DIR}/review_img.sh"

# Locate the profile image
if [[ -f "${SCRIPT_DIR}/me.jpeg" ]]; then
    PROFILE_IMG="${SCRIPT_DIR}/me.jpeg"
elif [[ -f "${WORKSPACE_ROOT}/me.jpeg" ]]; then
    PROFILE_IMG="${WORKSPACE_ROOT}/me.jpeg"
elif [[ -f "${WORKSPACE_ROOT}/generative_output/me.jpeg" ]]; then
    PROFILE_IMG="${WORKSPACE_ROOT}/generative_output/me.jpeg"
else
    echo "Error: Profile image me.jpeg not found!" >&2
    exit 1
fi

echo "Running visual review CLI on profile photo: ${PROFILE_IMG}"
echo "--------------------------------------------------------"
exec "${CLI}" "${PROFILE_IMG}"
