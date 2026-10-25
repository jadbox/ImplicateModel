#!/usr/bin/env bash
# review_video.sh - Forensic temporal and visual inspection of short video clips using local Qwen 3 VL on llama-server.
# Samples keyframes across the video duration and analyzes scene continuity, character trajectories, and temporal glitches.
# Usage: ./tools/review_video.sh [--frames <N>] [--timeout <SECONDS>] <VIDEO_FILE>

set -euo pipefail

# Configuration with environment variable overrides
DEFAULT_PORT="8935"
SERVER_HOST="${LLAMA_HOST:-127.0.0.1}"
SERVER_PORT="${LLAMA_PORT:-$DEFAULT_PORT}"
SERVER_URL="${LLAMA_SERVER_URL:-http://${SERVER_HOST}:${SERVER_PORT}}"
PARALLEL_SLOTS="${LLAMA_PARALLEL:-4}"
REQUEST_TIMEOUT="${LLAMA_TIMEOUT:-90}"
STARTUP_TIMEOUT="${LLAMA_STARTUP_TIMEOUT:-45}"
SAMPLE_FRAMES="5"

DEFAULT_MODEL="/mnt/storage/models/llama.cpp/Qwen3VL-8B-Instruct-Q4_K_M.gguf"
DEFAULT_MMPROJ="/mnt/storage/models/llama.cpp/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
MODEL_PATH="${LLAMA_MODEL_PATH:-$DEFAULT_MODEL}"
MMPROJ_PATH="${LLAMA_MMPROJ_PATH:-$DEFAULT_MMPROJ}"

# Lock and lifecycle directories
LOCK_DIR="/tmp/llama_vision"
ACTIVE_DIR="${LOCK_DIR}/active"
SERVER_PID_FILE="${LOCK_DIR}/server.pid"
SERVER_LOCK="${LOCK_DIR}/startup.lock"
MANAGED_FLAG="${LOCK_DIR}/managed.flag"

usage() {
    echo "Usage: $0 [--frames <N>] [--timeout <SECONDS>] <VIDEO_FILE>"
    echo "Example: $0 ./generative_output/pacman_generative_unfold.mp4"
    echo ""
    echo "Options:"
    echo "  --frames, -f <N>   Number of evenly spaced keyframes to sample (default: 5)"
    echo "  --timeout, -t <N>  Time limit for inference request in seconds (default: 90)"
    echo ""
    echo "Environment overrides:"
    echo "  LLAMA_TIMEOUT          Inference timeout in seconds (default: 90)"
    echo "  LLAMA_STARTUP_TIMEOUT  Server boot timeout in seconds (default: 45)"
    echo "  LLAMA_SERVER_URL       URL to llama-server (default: http://127.0.0.1:8935)"
    echo "  LLAMA_MODEL_PATH       Path to GGUF model"
    echo "  LLAMA_MMPROJ_PATH      Path to mmproj GGUF file"
    echo "  LLAMA_PARALLEL         Concurrent slot count (default: 4)"
    echo "  LLAMA_KEEP_ALIVE       If set to 1, do not shut down server after all jobs finish"
    exit 1
}

# Parse CLI arguments
VIDEO_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --frames|-f)
            SAMPLE_FRAMES="$2"
            shift 2
            ;;
        --timeout|-t)
            REQUEST_TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            if [[ -z "$VIDEO_PATH" ]]; then
                VIDEO_PATH="$1"
                shift
            else
                echo "Error: Unexpected argument '$1'" >&2
                usage
            fi
            ;;
    esac
done

if [[ -z "$VIDEO_PATH" ]]; then
    usage
fi

if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "Error: Video file not found: $VIDEO_PATH" >&2
    exit 1
fi

for cmd in ffmpeg ffprobe python3 curl; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: Required dependency '$cmd' is not installed or not in PATH." >&2
        exit 1
    fi
done

TMP_FRAMES_DIR="/tmp/opencode/video_review_${$}_${RANDOM}"
mkdir -p "$TMP_FRAMES_DIR"
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

# Reference-counted teardown: cleanup temp frames and stop server if last worker
cleanup() {
    rm -rf "$TMP_FRAMES_DIR" 2>/dev/null || true
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

# Extract video metadata and sample frames
python3 -c "
import sys, os, subprocess, json, urllib.request, base64

video_path = sys.argv[1]
tmp_dir = sys.argv[2]
sample_count = int(sys.argv[3])
server_url = sys.argv[4]
timeout_sec = int(sys.argv[5])

# Probe metadata
try:
    probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path]
    probe_data = json.loads(subprocess.check_output(probe_cmd))
    vstream = next(s for s in probe_data['streams'] if s['codec_type'] == 'video')
    w = int(vstream['width'])
    h = int(vstream['height'])
    duration = float(probe_data['format'].get('duration', vstream.get('duration', 1.0)))
    fps_eval = eval(vstream.get('r_frame_rate', '24/1')) if '/' in vstream.get('r_frame_rate', '') else float(vstream.get('r_frame_rate', 24))
except Exception as e:
    sys.exit(f'Failed to probe video: {e}')

# Extract frames
timestamps = [duration * (i + 0.5) / sample_count for i in range(sample_count)]
frame_files = []
for idx, t in enumerate(timestamps):
    out_file = os.path.join(tmp_dir, f'frame_{idx+1}.png')
    cmd = ['ffmpeg', '-y', '-ss', f'{t:.3f}', '-i', video_path, '-vframes', '1', out_file]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    frame_files.append((out_file, t))

# Prepare multimodal message payload
intro_text = (
    f'Video Metadata: duration={duration:.2f}s, resolution={w}x{h} ({w}px width, {h}px height), fps={fps_eval:.1f}.\n'
    f'Below is an evenly spaced sequence of {sample_count} sampled frames across the clip timeline.\n\n'
    'You are an expert video QA analyst and computer vision engineer conducting a forensic inspection of this short generative/rendered clip.\n'
    'Provide a thorough evaluation in structured Markdown covering:\n\n'
    '# 1. Video Overview & Scene Continuity\n'
    '- Overall environment, setting, background stability, camera regime (static, scrolling, panning), and global lighting/palette.\n'
    '- Bounding boxes [x1, y1, x2, y2] for significant persistent background or canvas regions.\n\n'
    '# 2. Characters, Motion Trajectories & Actions\n'
    '- Detail each character, creature, or primary moving object across the frame timeline.\n'
    '- For each entity, describe:\n'
    '  * Identity / Label\n'
    '  * Frame-by-frame action/motion path (direction, velocity, posture changes, mouth/limb animation)\n'
    '  * Bounding boxes [x1, y1, x2, y2] across sampled timestamps (where 0 <= x1 < x2 <= ' + str(w) + ' and 0 <= y1 < y2 <= ' + str(h) + ')\n\n'
    '# 3. Temporal Glitches, Inconsistencies & Visual Defects\n'
    'Scrutinize for video generation/rendering defects across time:\n'
    '- Shape morphing / flickering (characters unnaturally deforming or pulsating across frames)\n'
    '- Missing scene elements (e.g., absent background, missing maze walls, missing HUD/score)\n'
    '- Sprite fragmentation & black void clipping (broken parts, disconnected limbs, color bleed)\n'
    '- Motion discontinuity (teleporting, jumping without continuous trajectory)\n'
    '- Rate each defect with Severity: (Critical / High / Medium / Low / Clean)\n'
)

user_content = [{'type': 'text', 'text': intro_text}]
for idx, (fpath, t) in enumerate(frame_files):
    with open(fpath, 'rb') as fp:
        b64 = base64.b64encode(fp.read()).decode('utf-8')
    user_content.append({'type': 'text', 'text': f'--- Sampled Frame {idx+1} at t={t:.2f}s / {duration:.2f}s ---'})
    user_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}})

payload = {
    'model': 'default',
    'messages': [{'role': 'user', 'content': user_content}],
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
    sys.exit(f'Error: Video review inference timed out after {timeout_sec} seconds.')
except Exception as e:
    sys.exit(f'Video review request failed: {e}')
" "$VIDEO_PATH" "$TMP_FRAMES_DIR" "$SAMPLE_FRAMES" "$SERVER_URL" "$REQUEST_TIMEOUT"
