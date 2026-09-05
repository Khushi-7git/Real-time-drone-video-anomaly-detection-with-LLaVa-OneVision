# Real-Time Drone Anomaly Detection

Built for the **AHC Visual Intelligence Hackathon** — a one-day challenge on detecting anomalies in drone video in real time, cheaply enough to run across many live feeds at once.

## Problem

Drones flying over a city capture highways, streets, parks, and public spaces. Most of what they record is routine. A small fraction contains events that need a response — and the goal is to catch those events while the drone is still overhead, not in a later review of footage.

Standard object detectors (YOLO, classical CV) fall short here because **anomaly status depends on context, not object identity** — a stationary car is unremarkable in a parking lot and a problem on a highway shoulder. Vision-language models solve the context problem since they're queryable in language rather than tied to a fixed class list, but large VLMs are too slow and expensive to run continuously across many feeds.

**The core question this project explores:** can a *small* vision-language model do this reliably, in near real time?

### Events covered

- Traffic congestion
- Stalled / broken-down vehicle
- Vehicle blocking traffic
- Traffic accident
- Smoke
- Fire
- Waterlogging / flood
- Loitering / suspicious presence
- Road spill or debris
- Fighting / violence
- Wrong-way driving

## Approach

Two pipelines were built and compared:

### 1. LLaVA-OneVision zero-shot (primary)

No fine-tuning — inference only, to stay within a tight time budget and avoid the risk of a training run failing partway through.

```
Video → Frame Sampler → LLaVA-OneVision-0.5B (4-bit, zero-shot prompt)
      → Temporal Rule Layer → predictions.csv
```

- **Frame sampling** — a handful of frames pulled uniformly per video, turning "analyze a video" into "analyze a few images."
- **Zero-shot classification** — one prompt lists all event classes plus "normal"; the model returns a single class name per frame.
- **Temporal rule layer** — combines per-frame votes into one video-level label, based on how each event type actually unfolds:
  - **Instant classes** (`traffic_accident`, `fire`, `smoke`, `fighting_or_violence`) — a single hit on any frame triggers the alert immediately, since these events are over in about a second.
  - **Persistent classes** (`traffic_congestion`, `stalled_or_broken_down_vehicle`, `waterlogging_or_flood`, `loitering_or_suspicious_presence`, `road_spill_or_debris`) — must appear in ≥50% of sampled frames, since a single misread frame shouldn't trigger an alert for a condition that's supposed to persist.
  - This costs zero extra model calls and directly targets the brief's point that false alarms erode trust as much as missed detections do.

### 2. CLIP + LightGBM (backup / speed comparison)

```
Video → Frame Sampler → CLIP image encoder (embedding, no generation)
      → Mean-pool → LightGBM classifier → predictions.csv
```

Swaps autoregressive text generation for a single embedding pass per frame plus a lightweight trained classifier. Much faster per video than the VLM path, at the cost of losing the language-based flexibility to describe novel, unlisted event types.

## Repository structure

```
.
├── llava_onevision_fast_pipeline.py   # zero-shot LLaVA-OneVision pipeline
├── clip_lightgbm_fast_pipeline.py     # CLIP embeddings + LightGBM pipeline
├── anomaly_detection_deck.pptx        # presentation deck
└── README.md
```

Expected dataset layout:

```
dataset/
├── train/
│   └── <class_name>/
│       ├── videos/*.mp4
│       ├── videos.csv
│       └── ground_truth.csv
└── test/
    └── <class_name>/
        ├── videos/*.mp4
        ├── videos.csv
        └── ground_truth.csv
```

Unlabeled inference sets (no `ground_truth.csv`, flat `videos/` + `videos.csv` with a `filename`/`video_id` column) are also supported for prediction-only runs.

## Setup

```bash
pip install -q transformers accelerate bitsandbytes decord
# for the CLIP + LightGBM pipeline, also:
pip install -q scikit-learn lightgbm joblib
```

Both pipelines pull public model weights from Hugging Face (`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`, `openai/clip-vit-base-patch32`) — no API key or token required.

## Running

Edit `DATA_ROOT` at the top of either script to point at your dataset root, then:

```bash
python llava_onevision_fast_pipeline.py
python clip_lightgbm_fast_pipeline.py
```

Each writes a `*_predictions.csv` and prints video-level accuracy when ground truth is available.

### Config knobs

| Variable | Purpose |
|---|---|
| `NUM_FRAMES` | Frames sampled per video — lower is faster, less accurate |
| `MAX_VIDEOS_PER_CLASS` / `MAX_VIDEOS` | Caps total work for time-boxed runs |
| `PERSISTENCE_FRAC` | Vote threshold for persistent-class events (default 0.5) |
| `MODEL_ID` | Swap to a larger LLaVA-OneVision variant if time/VRAM allows |

## Known issue (fixed)

An early version of the zero-shot classifier decoded the model's **full output sequence**, including the input prompt — which itself lists every class name. This meant the decoded text always matched the first class in the list (`traffic_congestion`) regardless of what the model actually predicted, collapsing accuracy to near-random. The fix decodes only the newly generated tokens:

```python
input_len = inputs["input_ids"].shape[1]
generated_tokens = out[0][input_len:]
text = processor.decode(generated_tokens, skip_special_tokens=True).lower()
```

## Limitations

- **Zero-shot ceiling** — accuracy trails a fine-tuned model; this trade was made deliberately to stay inference-only under a tight time budget.
- **Reduced sampling** — small per-class video caps and frame counts (used for time-boxed demo runs) are not representative of the full dataset.
- **Not true streaming** — sampling from a completed file simplifies a live deployment, which would need a sliding window over incoming feed chunks.
- **Hand-set thresholds** — the instant/persistent split and 50% vote threshold are heuristics, not learned from data.

## Future work

- Run at full dataset scale once the pipeline is validated
- LoRA fine-tune on the same small model to close the zero-shot accuracy gap
- Learn event-duration thresholds from data instead of hand-setting them
- Extend to a sliding-window live-feed version for genuine real-time deployment
