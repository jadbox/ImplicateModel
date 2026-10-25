# `label_img.sh` — Visual Object & Behavior CSV Labeler

CLI tool for detecting and segmenting visual entities, active behaviors, and pixel bounding boxes directly into CSV format using the local `llama.cpp` server and **Qwen 3 VL (Q4 KM quants)**.

---

## Output Format

Each line of the output CSV contains exactly 6 comma-separated fields:

```text
object_names,behavior,x,y,x1,y1
```

- **`object_names`**: Pipe-delimited (`|`) aliases/synonyms for the detected object or character (e.g. `man|speaker|person`, `mario|plumber`, `suit|garment`, `piano`).
- **`behavior`**: Active verb, posture, or state (e.g. `speaking`, `gesturing`, `wearing`, `holding`, `standing`, `falling`, `running`).
- **`x,y`**: Top-left pixel coordinates (`x_min`, `y_min`).
- **`x1,y1`**: Bottom-right pixel coordinates (`x_max`, `y_max`), where `0 <= x < x1 <= width` and `0 <= y < y1 <= height`.

---

## Usage

```bash
# Label an image into a CSV file
./tools/label_img.sh <IMAGE_PATH> <OUTPUT_CSV>

# Label with an explicit 30s timeout cap
./tools/label_img.sh --timeout 30 <IMAGE_PATH> <OUTPUT_CSV>
```

### CLI Options & Environment Overrides

| Flag / Variable | Default | Purpose |
| :--- | :--- | :--- |
| `<IMAGE_PATH>` | (required) | Source image path (`.png`, `.jpg`, `.jpeg`, `.webp`) |
| `<OUTPUT_CSV>` | (required) | Destination path for generated CSV annotations |
| `--timeout`, `-t <N>` | `60` | CLI argument: Max seconds to wait for inference completion |
| `LLAMA_TIMEOUT` | `60` | Environment variable: Max seconds for inference request |
| `LLAMA_STARTUP_TIMEOUT`| `45` | Environment variable: Max seconds to wait for `llama-server` boot |
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8935` | Target llama.cpp server URL |
| `LLAMA_PORT` | `8935` | Server port |
| `LLAMA_PARALLEL` | `4` | Concurrent request slots |
| `LLAMA_MODEL_PATH` | `/mnt/storage/models/llama.cpp/Qwen3VL-8B-Instruct-Q4_K_M.gguf` | GGUF model path (Qwen 3 VL Q4_K_M) |
| `LLAMA_MMPROJ_PATH` | `/mnt/storage/models/llama.cpp/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf` | Multimodal projector path (Qwen 3 VL Q8_0) |
| `LLAMA_KEEP_ALIVE` | `0` | Set to `1` to prevent automatic teardown after requests finish |

---

## Examples & Real Outputs

### Example 1: Person / Portrait (`tools/me.jpeg`)

```bash
./tools/label_img.sh ./tools/me.jpeg ./output/me_labels.csv
```

**Output CSV (`./output/me_labels.csv`):**
```csv
man|speaker|person,gesturing,60,84,1196,505
suit|garment,draped,572,84,1196,505
shirt|garment,draped,572,204,1196,361
microphone|accessory,attached,920,344,960,361
curtain|scene_object,stationary,0,0,1200,722
```

### Example 2: Sprite Contact Sheet (`generative_output/smb3_mario_contact_sheet.png`)

```bash
./tools/label_img.sh ./generative_output/smb3_mario_contact_sheet.png ./output/mario_labels.csv
```

**Output CSV (`./output/mario_labels.csv`):**
```csv
mario|character,standing,32,24,160,232
mario|character,standing,232,24,360,232
cap|accessory,draped,48,24,144,64
overalls|garment,draped,48,112,144,192
boots|garment,draped,48,192,144,232
```

---

## Concurrency & Lifecycle

`label_img.sh` integrates seamlessly with `review_img.sh`:
- **Shared Mutex (`flock`)**: Prevents race conditions during cold starts.
- **Reference-Counted Server Management**: Tracks active requests and automatically terminates the managed server when the last worker finishes.
- **Pre-Existing Server Safety**: If `llama-server` is already active (e.g., from another thread or manual instance), it will not be terminated.
