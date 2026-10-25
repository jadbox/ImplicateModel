#!/usr/bin/env bash
# review_img.sh - Detailed visual inspection, trait analysis, and glitch review using local Qwen Vision on llama-server.
# Supports parallel executions with reference-counted server lifecycle, kernel flock mutex, and stale-lease pruning.
# Usage: ./tools/review_img.sh [--timeout <SECONDS>] <IMAGE_FILE>

set -euo pipefail

# Configuration with environment variable overrides
DEFAULT_PORT="8935"
SERVER_HOST="${LLAMA_HOST:-127.0.0.1}"
SERVER_PORT="${LLAMA_PORT:-$DEFAULT_PORT}"
SERVER_URL="${LLAMA_SERVER_URL:-http://${SERVER_HOST}:${SERVER_PORT}}"
PARALLEL_SLOTS="${LLAMA_PARALLEL:-4}"
REQUEST_TIMEOUT="${LLAMA_TIMEOUT:-60}"
STARTUP_TIMEOUT="${LLAMA_STARTUP_TIMEOUT:-45}"

DEFAULT_MODEL="/mnt/storage/models/llama.cpp/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
DEFAULT_MMPROJ="/mnt/storage/models/llama.cpp/mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf"
MODEL_PATH="${LLAMA_MODEL_PATH:-$DEFAULT_MODEL}"
MMPROJ_PATH="${LLAMA_MMPROJ_PATH:-$DEFAULT_MMPROJ}"

# Lock and lifecycle directories
LOCK_DIR="/tmp/llama_vision"
ACTIVE_DIR="${LOCK_DIR}/active"
SERVER_PID_FILE="${LOCK_DIR}/server.pid"
SERVER_LOCK="${LOCK_DIR}/startup.lock"
MANAGED_FLAG="${LOCK_DIR}/managed.flag"

usage() {
    echo "Usage: $0 [--timeout <SECONDS>] <IMAGE_FILE>"
    echo "Example: $0 --timeout 30 ./generative_output/nimh_brisby_contact_sheet.png"
    echo ""
    echo "Options:"
    echo "  --timeout, -t <N>  Time limit for inference request in seconds (default: 60)"
    echo ""
    echo "Environment overrides:"
    echo "  LLAMA_TIMEOUT          Inference timeout in seconds (default: 60)"
    echo "  LLAMA_STARTUP_TIMEOUT  Server boot timeout in seconds (default: 45)"
    echo "  LLAMA_SERVER_URL       URL to llama-server (default: http://127.0.0.1:8935)"
    echo "  LLAMA_MODEL_PATH       Path to GGUF model"
    echo "  LLAMA_MMPROJ_PATH      Path to mmproj GGUF file"
    echo "  LLAMA_PARALLEL         Concurrent slot count (default: 4)"
    echo "  LLAMA_KEEP_ALIVE       If set to 1, do not shut down server after all jobs finish"
    exit 1
}

# Parse CLI arguments
IMAGE_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout|-t)
            REQUEST_TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            if [[ -z "$IMAGE_PATH" ]]; then
                IMAGE_PATH="$1"
                shift
            else
                echo "Error: Unexpected argument '$1'" >&2
                usage
            fi
            ;;
    esac
done

if [[ -z "$IMAGE_PATH" ]]; then
    usage
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
    echo "Error: Image file not found: $IMAGE_PATH" >&2
    exit 1
fi

mkdir -p "$ACTIVE_DIR"
CLIENT_ID="client_${$}_${RANDOM}"
touch "${ACTIVE_DIR}/${CLIENT_ID}"

# Count active clients while self-healing stale leases from terminated/killed PIDs
get_active_count() {
    local count=0
    for f in "${ACTIVE_DIR}"/client_*; do
        [[ -e "$f" ]] || continue
        local cpid
        cpid=$(basename "$f" | cut -d'_' -f2)
        if [[ -n "$cpid" ]] && kill -0 "$cpid" 2>/dev/null; then
            count=$((count + 1))
        else
            # Stale client from an abrupt kill -9, prune it
            rm -f "$f" 2>/dev/null || true
        fi
    done
    echo "$count"
}

# Reference-counted teardown: only shutdown if all requests finished and server was launched by this CLI
cleanup() {
    rm -f "${ACTIVE_DIR}/${CLIENT_ID}" 2>/dev/null || true
    if [[ "${LLAMA_KEEP_ALIVE:-0}" == "1" ]]; then
        return 0
    fi
    (
        if flock -x -w 15 200; then
            local active_count
            active_count=$(get_active_count)
            if [[ "$active_count" -eq 0 && -f "$MANAGED_FLAG" ]]; then
                if [[ -f "$SERVER_PID_FILE" ]]; then
                    local spid
                    spid=$(cat "$SERVER_PID_FILE" 2>/dev/null || true)
                    if [[ -n "$spid" ]] && kill -0 "$spid" 2>/dev/null; then
                        echo "All review requests completed. Shutting down managed llama-server (PID $spid)..." >&2
                        kill "$spid" 2>/dev/null || true
                        for _ in {1..30}; do
                            kill -0 "$spid" 2>/dev/null || break
                            sleep 0.1
                        done
                        kill -9 "$spid" 2>/dev/null || true
                    fi
                    rm -f "$SERVER_PID_FILE" "$MANAGED_FLAG" 2>/dev/null || true
                fi
            fi
        fi
    ) 200>"${SERVER_LOCK}"
}
trap cleanup EXIT INT TERM

# Function to check server health
check_server() {
    local status
    status=$(curl -s --connect-timeout 2 "${SERVER_URL}/health" 2>/dev/null || true)
    if echo "$status" | grep -q '"status":"ok"'; then
        return 0
    fi
    return 1
}

# Auto-start server with atomic flock if not already running
if ! check_server; then
    (
        # Acquire lock with 15s timeout failsafe
        if ! flock -x -w 15 200; then
            echo "Error: Timed out waiting for startup lock ${SERVER_LOCK}" >&2
            exit 1
        fi

        # Re-check server status once lock is acquired
        if ! check_server; then
            if [[ -f "$MODEL_PATH" && -f "$MMPROJ_PATH" ]]; then
                echo "Notice: Starting shared llama-server on port ${SERVER_PORT} (slots=${PARALLEL_SLOTS})..." >&2
                touch "$MANAGED_FLAG"
                nohup llama-server \
                    -m "$MODEL_PATH" \
                    --mmproj "$MMPROJ_PATH" \
                    --image-min-tokens 1024 \
                    --cont-batching \
                    --parallel "$PARALLEL_SLOTS" \
                    -ngl 999 \
                    -fa on \
                    -c 32768 \
                    --port "$SERVER_PORT" \
                    --host "$SERVER_HOST" > /tmp/llama-server-vision.log 2>&1 200>&- &
                echo $! > "$SERVER_PID_FILE"
            else
                echo "Error: Server is down and model weights not found at $MODEL_PATH" >&2
                exit 1
            fi
        fi
    ) 200>"${SERVER_LOCK}"

    # Wait for server ready with configurable timeout
    echo -n "Waiting for model to load (limit: ${STARTUP_TIMEOUT}s)" >&2
    for ((i=1; i<=STARTUP_TIMEOUT; i++)); do
        if check_server; then
            echo " Ready!" >&2
            break
        fi
        echo -n "." >&2
        sleep 1
    done

    if ! check_server; then
        echo "" >&2
        echo "Error: Server failed to start within ${STARTUP_TIMEOUT}s limit. Check /tmp/llama-server-vision.log" >&2
        exit 1
    fi
fi

# Inspect image dimensions and MIME type using python3
READ_INFO=$(python3 -c "
import sys, os
from PIL import Image

path = sys.argv[1]
try:
    im = Image.open(path)
    w, h = im.size
    fmt = (im.format or 'png').lower()
    if fmt == 'jpg': fmt = 'jpeg'
    mime = f'image/{fmt}'
except Exception as e:
    sys.exit(f'Failed to open image: {e}')

print(f'{w}|{h}|{mime}')
sys.stdout.flush()
" "$IMAGE_PATH")

WIDTH=$(echo "$READ_INFO" | cut -d'|' -f1)
HEIGHT=$(echo "$READ_INFO" | cut -d'|' -f2)
MIME=$(echo "$READ_INFO" | cut -d'|' -f3)

# Carefully crafted prompt for detailed trait inspection, glitches, and pixel bounding boxes
PROMPT_TEXT="Image size: ${WIDTH}x${HEIGHT} (width=${WIDTH}px, height=${HEIGHT}px).

You are a critical computer-vision quality analyst inspecting rendered frames, sprites, and photographs for character/scene consistency, extraction bugs, and visual anomalies.

Provide a thorough, highly specific evaluation covering:

1. SCENE & LAYOUT:
   - Describe background, composition, panels/splits, and overall setup. Include bounding boxes [x1, y1, x2, y2].

2. CHARACTER & ITEM TRAITS:
   - Identify every character, person, creature, costume part, held item, or key visual element.
   - For EACH individual element/trait (e.g. face/head, hair, clothing, lapels/shirt, hands/gestures, held staff/tool, accessories/mic, cutout patches), detail:
     * Label / Trait name
     * Detailed appearance (colors, materials, pose, anatomy, style)
     * Pixel bounding box: [x1, y1, x2, y2] where 0 <= x1 < x2 <= ${WIDTH} and 0 <= y1 < y2 <= ${HEIGHT}.

3. VISUAL ANOMALIES & GLITCHES:
   - Look intensely for defects: missing body parts, erased facial features, mask cutout failures, severe clipping, color corruption, jagged boundaries, incomplete strokes, fragmented items, artifacts, watermark/logos, noise floor corruption, or split lines.
   - For EACH defect or artifact found, detail:
     * Glitch Type / Title
     * Detailed Description of what is visually broken or corrupted
     * Location / Pixel Bounding Box: [x1, y1, x2, y2]
     * Severity: (Critical / High / Medium / Low / Minor Overlay)
   - If no rendering errors are present, state that the image is clean and note any incidental artifacts (e.g. logos or watermarks).

Be precise and observant. Every detected trait and glitch must have a [x1, y1, x2, y2] pixel box."

# Run inference via Python with explicit timeout
python3 -c "
import sys, urllib.request, json, base64

image_path = sys.argv[1]
server_url = sys.argv[2]
mime = sys.argv[3]
prompt = sys.argv[4]
timeout_sec = int(sys.argv[5])

with open(image_path, 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('utf-8')

payload = {
    'model': 'default',
    'messages': [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{b64}'}}
            ]
        }
    ],
    'max_tokens': 2048,
    'temperature': 0.1
}

req = urllib.request.Request(
    f'{server_url}/v1/chat/completions',
    data=json.dumps(payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

try:
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        content = res['choices'][0]['message']['content']
        print(content)
except TimeoutError:
    sys.exit(f'Error: Inference request timed out after {timeout_sec} seconds.')
except Exception as e:
    sys.exit(f'Inference request failed: {e}')
" "$IMAGE_PATH" "$SERVER_URL" "$MIME" "$PROMPT_TEXT" "$REQUEST_TIMEOUT"
