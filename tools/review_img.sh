#!/usr/bin/env bash
# review_img.sh - Detailed visual inspection, trait analysis, and glitch review using local Qwen Vision on llama-server.
# Usage: ./tools/review_img.sh <IMAGE_FILE>

set -euo pipefail

# Configuration with environment variable overrides
DEFAULT_PORT="8935"
SERVER_HOST="${LLAMA_HOST:-127.0.0.1}"
SERVER_PORT="${LLAMA_PORT:-$DEFAULT_PORT}"
SERVER_URL="${LLAMA_SERVER_URL:-http://${SERVER_HOST}:${SERVER_PORT}}"

DEFAULT_MODEL="/mnt/storage/models/llama.cpp/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
DEFAULT_MMPROJ="/mnt/storage/models/llama.cpp/mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf"
MODEL_PATH="${LLAMA_MODEL_PATH:-$DEFAULT_MODEL}"
MMPROJ_PATH="${LLAMA_MMPROJ_PATH:-$DEFAULT_MMPROJ}"

usage() {
    echo "Usage: $0 <IMAGE_FILE>"
    echo "Example: $0 ./generative_output/nimh_brisby_contact_sheet.png"
    echo ""
    echo "Environment overrides:"
    echo "  LLAMA_SERVER_URL  URL to llama-server (default: http://127.0.0.1:8935)"
    echo "  LLAMA_MODEL_PATH  Path to GGUF model"
    echo "  LLAMA_MMPROJ_PATH Path to mmproj GGUF file"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

IMAGE_PATH="$1"

if [[ ! -f "$IMAGE_PATH" ]]; then
    echo "Error: Image file not found: $IMAGE_PATH" >&2
    exit 1
fi

# Function to check server health
check_server() {
    local status
    status=$(curl -s --connect-timeout 2 "${SERVER_URL}/health" 2>/dev/null || true)
    if echo "$status" | grep -q '"status":"ok"'; then
        return 0
    fi
    return 1
}

# Auto-start server if not already running
if ! check_server; then
    echo "Notice: llama-server not responding at ${SERVER_URL}." >&2
    if [[ -f "$MODEL_PATH" && -f "$MMPROJ_PATH" ]]; then
        echo "Starting llama-server with Qwen Vision model on port ${SERVER_PORT}..." >&2
        nohup llama-server \
            -m "$MODEL_PATH" \
            --mmproj "$MMPROJ_PATH" \
            -ngl 999 \
            -fa on \
            -c 16384 \
            --port "$SERVER_PORT" \
            --host "$SERVER_HOST" > /tmp/llama-server-vision.log 2>&1 &
        
        # Wait for server ready
        echo -n "Waiting for model to load" >&2
        for _ in {1..30}; do
            if check_server; then
                echo " Ready!" >&2
                break
            fi
            echo -n "." >&2
            sleep 1
        done
        
        if ! check_server; then
            echo "Error: Server failed to start within 30s. Check /tmp/llama-server-vision.log" >&2
            exit 1
        fi
    else
        echo "Error: Server is down and model weights not found at $MODEL_PATH" >&2
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

# Run inference via Python
python3 -c "
import sys, urllib.request, json, base64

image_path = sys.argv[1]
server_url = sys.argv[2]
mime = sys.argv[3]
prompt = sys.argv[4]

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
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        content = res['choices'][0]['message']['content']
        print(content)
except Exception as e:
    sys.exit(f'Inference request failed: {e}')
" "$IMAGE_PATH" "$SERVER_URL" "$MIME" "$PROMPT_TEXT"
