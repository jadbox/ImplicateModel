#!/usr/bin/env bash
# label_img.sh - Segment, detect, and label visual objects, behaviors, and pixel bounding boxes into CSV format.
# Powered by local llama.cpp server and Qwen 3 VL (Q4 KM quants).
# Usage: ./tools/label_img.sh [--timeout <SECONDS>] <IMAGE_FILE> <OUTPUT_CSV_FILE>

set -euo pipefail

# Configuration with environment variable overrides
DEFAULT_PORT="8935"
SERVER_HOST="${LLAMA_HOST:-127.0.0.1}"
SERVER_PORT="${LLAMA_PORT:-$DEFAULT_PORT}"
SERVER_URL="${LLAMA_SERVER_URL:-http://${SERVER_HOST}:${SERVER_PORT}}"
PARALLEL_SLOTS="${LLAMA_PARALLEL:-4}"
REQUEST_TIMEOUT="${LLAMA_TIMEOUT:-60}"
STARTUP_TIMEOUT="${LLAMA_STARTUP_TIMEOUT:-45}"

DEFAULT_MODEL="/mnt/storage/models/llama.cpp/Qwen3VL-8B-Instruct-Q4_K_M.gguf"
DEFAULT_MMPROJ="/mnt/storage/models/llama.cpp/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
MODEL_PATH="${LLAMA_MODEL_PATH:-$DEFAULT_MODEL}"
MMPROJ_PATH="${LLAMA_MMPROJ_PATH:-$DEFAULT_MMPROJ}"

# Shared lock and lifecycle directories
LOCK_DIR="/tmp/llama_vision"
ACTIVE_DIR="${LOCK_DIR}/active"
SERVER_PID_FILE="${LOCK_DIR}/server.pid"
SERVER_LOCK="${LOCK_DIR}/startup.lock"
MANAGED_FLAG="${LOCK_DIR}/managed.flag"

usage() {
    echo "Usage: $0 [--timeout <SECONDS>] <IMAGE_FILE> <OUTPUT_CSV_FILE>"
    echo "Example: $0 ./generative_output/smb3_mario_contact_sheet.png ./labels.csv"
    echo ""
    echo "Output CSV format: object_names,behavior,x,y,x1,y1"
    echo "  e.g.: man|speaker|person,gesturing,60,7,1080,650"
    echo "        piano,falling,10,10,200,240"
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
OUTPUT_CSV=""
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
            elif [[ -z "$OUTPUT_CSV" ]]; then
                OUTPUT_CSV="$1"
                shift
            else
                echo "Error: Unexpected argument '$1'" >&2
                usage
            fi
            ;;
    esac
done

if [[ -z "$IMAGE_PATH" || -z "$OUTPUT_CSV" ]]; then
    usage
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
    echo "Error: Image file not found: $IMAGE_PATH" >&2
    exit 1
fi

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_CSV")"

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
                        echo "All requests completed. Shutting down managed llama-server (PID $spid)..." >&2
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
        if ! flock -x -w 15 200; then
            echo "Error: Timed out waiting for startup lock ${SERVER_LOCK}" >&2
            exit 1
        fi

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

# Carefully crafted prompt for entity detection, behaviors, and bounding boxes
PROMPT_TEXT="Image size: ${WIDTH}x${HEIGHT} (width=${WIDTH}px, height=${HEIGHT}px).

You are an expert visual object detector and segmenter.
Locate and label all distinct characters, persons, creatures, garments, accessories, held items, and scene objects in this image.

CRITICAL BOUNDING BOX REQUIREMENT:
Each bounding box MUST be a tight, localized box [ymin, xmin, ymax, xmax] around that specific object.
DO NOT output full-frame boxes [0, 0, 1000, 1000] for individual objects.

Format each line as:
object_names,behavior,ymin,xmin,ymax,xmax

- object_names: comma-free, pipe-separated names/aliases (e.g. 'man|plumber|mario', 'mouse|creature|mrs_brisby', 'staff|walking_stick', 'cloak|cape', 'piano')
- behavior: active action, state, or posture (e.g. 'running', 'falling', 'speaking', 'gesturing', 'holding_staff', 'standing', 'draped')
- ymin,xmin,ymax,xmax: normalized integers in the 0 to 1000 range.

Output ONLY CSV lines. Do not wrap in markdown or backticks."

# Run inference and write clean CSV
python3 -c "
import sys, urllib.request, json, base64

image_path = sys.argv[1]
server_url = sys.argv[2]
mime = sys.argv[3]
prompt = sys.argv[4]
timeout_sec = int(sys.argv[5])
output_csv = sys.argv[6]
w = int(sys.argv[7])
h = int(sys.argv[8])

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
    'max_tokens': 1500,
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
        raw = res['choices'][0]['message']['content'].strip()
except TimeoutError:
    sys.exit(f'Error: Inference request timed out after {timeout_sec} seconds.')
except Exception as e:
    sys.exit(f'Inference request failed: {e}')

lines = []
for line in raw.splitlines():
    line = line.strip().strip('\`').strip()
    if not line or ',' not in line or line.startswith('object_names'):
        continue
    parts = [p.strip() for p in line.split(',')]
    if len(parts) >= 6:
        obj = parts[0]
        beh = parts[1]
        try:
            ymin, xmin, ymax, xmax = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            x = max(0, min(w, int(round(xmin * w / 1000.0))))
            y = max(0, min(h, int(round(ymin * h / 1000.0))))
            x1 = max(0, min(w, int(round(xmax * w / 1000.0))))
            y1 = max(0, min(h, int(round(ymax * h / 1000.0))))
            if x1 > x and y1 > y:
                lines.append(f'{obj},{beh},{x},{y},{x1},{y1}')
        except Exception:
            continue

with open(output_csv, 'w') as f:
    for l in lines:
        f.write(l + '\n')

print(f'Wrote {len(lines)} labeled segments to {output_csv}:')
for l in lines:
    print('  ' + l)
" "$IMAGE_PATH" "$SERVER_URL" "$MIME" "$PROMPT_TEXT" "$REQUEST_TIMEOUT" "$OUTPUT_CSV" "$WIDTH" "$HEIGHT"
