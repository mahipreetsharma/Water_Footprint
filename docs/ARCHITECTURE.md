# 🏗️ AquaLens — Deep Dive Architecture

## Overview

AquaLens is composed of four loosely-coupled layers. Data flows **top-to-bottom** from user input to rendered output. Each layer is independently testable.

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — INTERFACE LAYER                                   │
│  Streamlit (frontend.py)                                     │
│  • Renders UI tabs, widgets, charts                         │
│  • Manages session state & history                           │
│  • Language selection triggers translation lookup            │
└────────────────────────┬────────────────────────────────────┘
                         │ calls
┌────────────────────────▼────────────────────────────────────┐
│  LAYER 2 — PROCESSING LAYER                                  │
│                                                             │
│  ┌─────────────────────┐  ┌──────────────────────────────┐  │
│  │  vision_module.py   │  │  footprint_engine.py         │  │
│  │  • YOLOv8 inference │  │  • JSON database lookup      │  │
│  │  • Bounding boxes   │  │  • Fuzzy matching (rapidfuzz)│  │
│  │  • Top label pick   │  │  • Total water calculation   │  │
│  └─────────┬───────────┘  └────────────────┬─────────────┘  │
│            └──────────────────────────────┘                 │
│                          │                                   │
│                ┌─────────▼──────────────┐                   │
│                │  translation_utils.py  │                   │
│                │  • Load language JSON  │                   │
│                │  • key → string lookup │                   │
│                └─────────┬──────────────┘                   │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  LAYER 3 — DATA LAYER                                        │
│  • footprint_mapping.json  (product → water values)         │
│  • eco_tips.json           (category → tips list)           │
│  • translations/*.json     (lang_code → UI strings)         │
│  • model/weights/best.pt   (trained YOLO weights)           │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Dependency Graph

```
frontend.py
├── footprint_engine.py
│   ├── data/footprint_mapping.json
│   └── data/eco_tips.json
├── vision_module.py
│   └── model/weights/best.pt (or yolov8n.pt fallback)
├── translation_utils.py
│   └── data/translations/*.json
└── history_manager.py
    └── (streamlit session_state — in-memory only)
```

---

## Data Flow: Text Search Path

```
User types "cotton shirt"
    │
    ▼
footprint_engine.lookup("cotton shirt")
    │ normalize → "cotton_shirt"
    │ exact match found in footprint_mapping.json
    ▼
returns {green: 2500, blue: 200, grey: 50, unit: "per_item", ...}
    │
    ▼
render_footprint_card() builds Plotly chart + metric widgets
    │
    ▼
translation_utils.t("green_water", "hi") → "हरा जल"
    │
    ▼
history_manager.add_to_history("Cotton Shirt", data)
    │
    ▼
UI renders: card + chart + eco tip
```

## Data Flow: Image Upload Path

```
User uploads image.jpg
    │
    ▼
vision_module.detect_from_image(pil_image)
    │ YOLOv8 inference → [("cotton_shirt", 0.92), ("background", 0.45)]
    │ sort by confidence → top result: "cotton_shirt"
    ▼
footprint_engine.lookup("cotton_shirt")
    │ → footprint data
    ▼
(same as text search path from here)
```

---

## Fuzzy Matching Algorithm

`rapidfuzz.process.extractOne` with `token_sort_ratio`:

| Input | Matched Key | Score |
|---|---|---|
| `"cotton shirt"` | `cotton_shirt` | 100 |
| `"cottn shrt"` | `cotton_shirt` | ~85 |
| `"shirt cotton"` | `cotton_shirt` | 100 (token sort) |
| `"xyzgarbage123"` | (None, below 70) | — |

The threshold of **70** was chosen to allow typos while avoiding false positives.

---

## Multilingual i18n System

```
Translation files: data/translations/<code>.json
      en.json  ← master template (all keys must exist here)
      hi.json  ← subset or full match of en.json keys
      ta.json
      ...

translation_utils.t(key, lang_code):
  1. Load lang_code.json into memory cache
  2. Return translations[key]
  3. If key missing → return key itself (graceful degradation)
  4. If file missing → fall back to en.json
```

---

## YOLOv8 Model Details

| Parameter | Value |
|---|---|
| Base model | `yolov8s.pt` (small, 22MB) |
| Input size | 640×640 px |
| Training epochs | 50 (with early stopping at patience=15) |
| Output classes | 25–30 (matches footprint_mapping.json keys) |
| Inference device | CPU (default) or CUDA GPU |
| Confidence threshold | 0.40 |
| Weights path | `model/weights/best.pt` |

**Class Name Convention:** All YOLO class names must exactly match keys in `footprint_mapping.json`. Use `snake_case` (e.g., `cotton_shirt`, not `Cotton Shirt`).

---

## Session State Schema (Streamlit)

```python
st.session_state = {
    "scan_history": [
        {
            "timestamp": "2024-01-15 14:32",
            "product": "Cotton Shirt",
            "total_litres": 2750,
            "category": "clothing"
        },
        ...
    ]
}
```

History is **in-memory only** and resets on page refresh. For persistent storage, future versions could use SQLite or a backend API.

---

## Scaling Considerations

| Concern | Current Approach | Future Upgrade |
|---|---|---|
| Database size | JSON file (< 500 items) | SQLite / PostgreSQL |
| Image inference | Synchronous, per-request | Async job queue (Celery) |
| Translation | Static JSON files | LLM-backed auto-translate |
| History | Streamlit session state | User accounts + database |
| Camera | streamlit-webrtc (local) | WebSocket + TURN server |
| Deployment | `streamlit run` locally | Docker + Streamlit Cloud / HuggingFace Spaces |
