# `review_img.sh` — Visual QA & Defect Inspection CLI

Automated visual review CLI powered by local `llama.cpp` server and the Qwen 3 VL (Q4 KM quants) vision-language model.

---

## Capabilities

1. **Scene & Layout Analysis**
   - Backdrop, composition, framing, split-screen/contact-sheet detection.
   - Spatial coordinates for panels and scene regions.

2. **Character & Item Traits**
   - Fine-grained breakdown: head/face, hair, attire/fabrics, held objects/tools, accessories, posture, and gestures.
   - Dedicated pixel bounding box `[x1, y1, x2, y2]` for each trait.

3. **Visual Anomalies & Glitch Auditing**
   - Scrutinizes for generative/extraction flaws: missing anatomy/faces, mask cutout failures, jagged borders, black void clipping, disjointed/fragmented items, and divider line artifacts.
   - For clean photos, verifies visual fidelity and identifies incidental overlays (e.g. corner watermarks/logos).
   - Pixel bounding boxes and severity ratings (`Critical`, `High`, `Medium`, `Low`, `Minor Overlay`, `Clean`).

---

## Usage

```bash
# Review an image with default 60s timeout
./tools/review_img.sh <IMAGE_PATH>

# Review an image with a strict 30s timeout limit
./tools/review_img.sh --timeout 30 <IMAGE_PATH>

# Run example test on user profile photo
./tools/review_img_example.sh
```

### CLI Options & Environment Overrides

| Flag / Variable | Default | Purpose |
| :--- | :--- | :--- |
| `--timeout`, `-t <N>` | `60` | CLI argument: Max seconds to wait for inference completion |
| `LLAMA_TIMEOUT` | `60` | Environment variable: Max seconds for inference request |
| `LLAMA_STARTUP_TIMEOUT`| `45` | Environment variable: Max seconds to wait for `llama-server` boot |
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8935` | Target llama.cpp server URL |
| `LLAMA_PORT` | `8935` | Server port |
| `LLAMA_PARALLEL` | `4` | Concurrent request slots |
| `LLAMA_MODEL_PATH` | `/mnt/storage/models/llama.cpp/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | GGUF model path |
| `LLAMA_MMPROJ_PATH` | `/mnt/storage/models/llama.cpp/mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf` | Multimodal projector path |
| `LLAMA_KEEP_ALIVE` | `0` | Set to `1` to prevent automatic teardown after requests finish |


---

## Concurrency & Lifecycle Architecture

When run across multiple threads or parallel workers (e.g., `xargs -P 4`):

1. **Single-Instance Mutex (`flock`)**
   - Cold starts use an atomic lock (`/tmp/llama_vision/startup.lock`).
   - Only one process initializes `llama-server`. Concurrent workers wait on the lock and poll `/health` until the model is loaded, preventing duplicate VRAM allocations or port collisions.

2. **Continuous Batching & Multi-Slot Parallelism**
   - Server launches with `--parallel 4`, `--cont-batching`, and `--image-min-tokens 1024`.
   - Parallel inference requests execute concurrently across available slots.

3. **Reference-Counted Request Tracking (`/tmp/llama_vision/active/`)**
   - Each process creates an active lease (`client_${PID}_${RANDOM}`) upon entry.
   - An `EXIT INT TERM` trap ensures the token is unlinked upon completion.

4. **Reference-Counted Teardown**
   - When each worker finishes, it checks the active request count.
   - If `active_count == 0` and the server was started by this CLI (`managed.flag`), the final worker terminates `llama-server`.
   - **External Server Protection:** If `llama-server` was already running before the script started (e.g., manual daemon or other test suite), it is marked unmanaged and will **not** be killed.

---

## Lock Failsafes & Resilience

### 1. What happens if a process crashes while holding the lock?
`flock(2)` is managed directly by the Linux kernel on the open file descriptor:
- If a process crashes, encounters an unhandled error, exits, or receives `SIGKILL` (`kill -9`), the kernel **automatically closes all open file descriptors** for that process.
- Closing the file descriptor **instantly releases the kernel lock**. An advisory `flock` cannot outlive the process that acquired it.

### 2. Lock Acquisition Timeout (`-w 15`)
- Both startup and teardown lock acquisitions use `flock -x -w 15`.
- If an anomalous process hangs while holding the lock, waiting processes will time out after 15 seconds rather than blocking indefinitely.

### 3. Self-Healing Stale Lease Pruning
- If a client process is forcefully killed (`kill -9`), bash traps do not run, leaving a lease file in `/tmp/llama_vision/active/`.
- The active client counter performs a liveness probe (`kill -0 $PID`) on every check. Dead PIDs are immediately purged, ensuring the server teardown counter cannot be locked by zombie leases.
