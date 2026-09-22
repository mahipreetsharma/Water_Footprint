# 🤖 AGENT_GUIDE.md — AquaLens Build Instructions for AI Agents

> **Read this file first.** This document is a self-contained, step-by-step guide for an AI coding agent to build the entire AquaLens project from scratch. Follow each phase in order. Do not skip phases.

---

## 📋 Project Summary (30-second brief)

**Goal:** Build a Python + Streamlit web app called AquaLens that:
1. Accepts user input via text search, image upload, or live camera scan
2. Identifies everyday products using a YOLOv8 computer vision model
3. Returns a water footprint breakdown (Green / Blue / Grey water in litres)
4. Displays the result in a multilingual UI (English + 6 Indian languages)

**Primary stack:** Python 3.9+, Streamlit, Ultralytics YOLOv8, OpenCV, Plotly, rapidfuzz

---

## 🗂️ Repository Layout You Must Create

```
AquaLens/
├── data/
│   ├── footprint_mapping.json
│   ├── eco_tips.json
│   └── translations/
│       ├── en.json  hi.json  ta.json  te.json  bn.json  mr.json  kn.json
├── model/
│   ├── train_detector.py
│   ├── predict.py
│   └── weights/   (gitignored — populated after training)
├── app/
│   ├── frontend.py
│   ├── footprint_engine.py
│   ├── vision_module.py
│   ├── translation_utils.py
│   ├── pdf_exporter.py
│   ├── history_manager.py
│   └── assets/  (style.css, logo placeholder)
├── tests/
│   ├── test_footprint_engine.py
│   └── test_translation_utils.py
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Phase 0 — Environment & Dependencies

### Step 0.1 — Create `requirements.txt`

```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
streamlit>=1.35.0
streamlit-webrtc>=0.47.0
Pillow>=10.0.0
plotly>=5.0.0
pandas>=2.0.0
rapidfuzz>=3.0.0
fpdf2>=2.7.0
python-dotenv>=1.0.0
```

### Step 0.2 — Create `requirements-dev.txt`

```
pytest>=8.0.0
pytest-cov>=5.0.0
black>=24.0.0
ruff>=0.4.0
```

### Step 0.3 — Create `.gitignore`

```
venv/
__pycache__/
*.pyc
.env
model/weights/
*.pt
runs/
data/yolo_dataset/
*.egg-info/
dist/
.DS_Store
```

### Step 0.4 — Create `.env.example`

```
# Optional: set to a custom port
STREAMLIT_PORT=8501

# Optional: override default language
DEFAULT_LANG=en
```

---

## 📊 Phase 1 — Build the Data Layer

### Step 1.1 — Create `data/footprint_mapping.json`

This is the core database. Each key is the YOLO detection label (snake_case). Populate with at least the following 25 entries. Values are in **litres**.

```json
{
  "cotton_shirt": {
    "green_water_litres": 2500,
    "blue_water_litres": 200,
    "grey_water_litres": 50,
    "unit": "per_item",
    "category": "clothing",
    "source": "Water Footprint Network"
  },
  "jeans": {
    "green_water_litres": 6800,
    "blue_water_litres": 800,
    "grey_water_litres": 400,
    "unit": "per_item",
    "category": "clothing",
    "source": "Water Footprint Network"
  },
  "apple": {
    "green_water_litres": 600,
    "blue_water_litres": 80,
    "grey_water_litres": 30,
    "unit": "per_kg",
    "category": "food",
    "source": "Water Footprint Network"
  },
  "banana": {
    "green_water_litres": 700,
    "blue_water_litres": 40,
    "grey_water_litres": 20,
    "unit": "per_kg",
    "category": "food",
    "source": "Water Footprint Network"
  },
  "rice": {
    "green_water_litres": 1100,
    "blue_water_litres": 450,
    "grey_water_litres": 80,
    "unit": "per_kg",
    "category": "food",
    "source": "Water Footprint Network"
  },
  "wheat_bread": {
    "green_water_litres": 900,
    "blue_water_litres": 140,
    "grey_water_litres": 60,
    "unit": "per_kg",
    "category": "food",
    "source": "Water Footprint Network"
  },
  "beef": {
    "green_water_litres": 13200,
    "blue_water_litres": 550,
    "grey_water_litres": 1650,
    "unit": "per_kg",
    "category": "food",
    "source": "Water Footprint Network"
  },
  "chicken": {
    "green_water_litres": 3000,
    "blue_water_litres": 220,
    "grey_water_litres": 130,
    "unit": "per_kg",
    "category": "food",
    "source": "Water Footprint Network"
  },
  "coffee_cup": {
    "green_water_litres": 120,
    "blue_water_litres": 15,
    "grey_water_litres": 5,
    "unit": "per_cup",
    "category": "beverage",
    "source": "Water Footprint Network"
  },
  "tea_cup": {
    "green_water_litres": 25,
    "blue_water_litres": 5,
    "grey_water_litres": 2,
    "unit": "per_cup",
    "category": "beverage",
    "source": "Water Footprint Network"
  },
  "milk": {
    "green_water_litres": 800,
    "blue_water_litres": 80,
    "grey_water_litres": 30,
    "unit": "per_litre",
    "category": "beverage",
    "source": "Water Footprint Network"
  },
  "beer": {
    "green_water_litres": 250,
    "blue_water_litres": 35,
    "grey_water_litres": 15,
    "unit": "per_litre",
    "category": "beverage",
    "source": "Water Footprint Network"
  },
  "smartphone": {
    "green_water_litres": 0,
    "blue_water_litres": 10000,
    "grey_water_litres": 2500,
    "unit": "per_item",
    "category": "electronics",
    "source": "Estimated from semiconductor manufacturing data"
  },
  "laptop": {
    "green_water_litres": 0,
    "blue_water_litres": 20000,
    "grey_water_litres": 5000,
    "unit": "per_item",
    "category": "electronics",
    "source": "Estimated from semiconductor manufacturing data"
  },
  "egg": {
    "green_water_litres": 180,
    "blue_water_litres": 15,
    "grey_water_litres": 10,
    "unit": "per_item",
    "category": "food",
    "source": "Water Footprint Network"
  },
  "potato": {
    "green_water_litres": 170,
    "blue_water_litres": 30,
    "grey_water_litres": 50,
    "unit": "per_kg",
    "category": "food",
    "source": "Water Footprint Network"
  },
  "tomato": {
    "green_water_litres": 160,
    "blue_water_litres": 20,
    "grey_water_litres": 20,
    "unit": "per_kg",
    "category": "food",
    "source": "Water Footprint Network"
  },
  "sugar": {
    "green_water_litres": 1300,
    "blue_water_litres": 80,
    "grey_water_litres": 50,
    "unit": "per_kg",
    "category": "food",
    "source": "Water Footprint Network"
  },
  "leather_shoes": {
    "green_water_litres": 7000,
    "blue_water_litres": 600,
    "grey_water_litres": 300,
    "unit": "per_pair",
    "category": "clothing",
    "source": "Estimated from leather tanning data"
  },
  "paper_sheet": {
    "green_water_litres": 10,
    "blue_water_litres": 2,
    "grey_water_litres": 1,
    "unit": "per_sheet",
    "category": "stationery",
    "source": "Water Footprint Network"
  },
  "orange": {
    "green_water_litres": 450,
    "blue_water_litres": 60,
    "grey_water_litres": 30,
    "unit": "per_kg",
    "category": "food",
    "source": "Water Footprint Network"
  },
  "mango": {
    "green_water_litres": 680,
    "blue_water_litres": 150,
    "grey_water_litres": 40,
    "unit": "per_kg",
    "category": "food",
    "source": "Water Footprint Network"
  },
  "chocolate_bar": {
    "green_water_litres": 1700,
    "blue_water_litres": 100,
    "grey_water_litres": 60,
    "unit": "per_100g",
    "category": "food",
    "source": "Water Footprint Network"
  },
  "plastic_bottle": {
    "green_water_litres": 0,
    "blue_water_litres": 6,
    "grey_water_litres": 2,
    "unit": "per_item",
    "category": "packaging",
    "source": "Estimated from PET production data"
  },
  "denim_jacket": {
    "green_water_litres": 9000,
    "blue_water_litres": 1000,
    "grey_water_litres": 500,
    "unit": "per_item",
    "category": "clothing",
    "source": "Water Footprint Network"
  }
}
```

### Step 1.2 — Create `data/eco_tips.json`

```json
{
  "food": [
    "Choose plant-based meals to reduce your water footprint by up to 50%.",
    "Avoid food waste — wasted food means wasted water.",
    "Buy locally grown produce to cut water used in transport cooling."
  ],
  "clothing": [
    "Buy second-hand clothes to save thousands of litres of water.",
    "Wash clothes in cold water and only when full.",
    "Choose organic cotton — it uses less blue water than conventional cotton."
  ],
  "beverage": [
    "Drink tap water instead of bottled beverages when safe.",
    "Switch from coffee to tea — tea has 10x lower water footprint.",
    "Reduce alcohol consumption to save significant blue water."
  ],
  "electronics": [
    "Repair electronics instead of replacing them.",
    "Donate old devices to extend their useful life.",
    "Choose refurbished electronics over new ones."
  ],
  "default": [
    "Every litre saved counts — small changes create big impact.",
    "Share your water footprint results to raise community awareness."
  ]
}
```

### Step 1.3 — Create `data/translations/en.json` (English — Master template)

All other language files must match these exact keys.

```json
{
  "app_title": "AquaLens - Water Footprint Estimator",
  "app_tagline": "Scan any product. Know its hidden water cost.",
  "tab_text_search": "Text Search",
  "tab_image_upload": "Image Upload",
  "tab_camera_scan": "Camera Scan",
  "tab_history": "My History",
  "search_placeholder": "Type a product name (e.g. cotton shirt, apple, coffee)...",
  "search_button": "Search",
  "scan_button": "Start Scan",
  "upload_button": "Upload Image",
  "result_title": "Water Footprint for",
  "green_water": "Green Water",
  "blue_water": "Blue Water",
  "grey_water": "Grey Water",
  "total_water": "Total Water",
  "unit_label": "Measurement unit",
  "eco_tip_header": "💡 Eco Tip",
  "export_pdf": "Export as PDF",
  "no_result": "Product not found. Try a different name.",
  "history_header": "Your Water Footprint History",
  "history_empty": "No scans yet. Start scanning products!",
  "language_label": "Language",
  "source_label": "Data Source",
  "category_label": "Category",
  "confidence_label": "Detection Confidence"
}
```

### Step 1.4 — Create `data/translations/hi.json` (Hindi)

```json
{
  "app_title": "AquaLens - जल पदचिह्न अनुमानक",
  "app_tagline": "किसी भी उत्पाद को स्कैन करें। उसकी छिपी जल लागत जानें।",
  "tab_text_search": "टेक्स्ट खोज",
  "tab_image_upload": "छवि अपलोड",
  "tab_camera_scan": "कैमरा स्कैन",
  "tab_history": "मेरा इतिहास",
  "search_placeholder": "उत्पाद का नाम टाइप करें (जैसे कपास शर्ट, सेब, कॉफी)...",
  "search_button": "खोजें",
  "scan_button": "स्कैन शुरू करें",
  "upload_button": "छवि अपलोड करें",
  "result_title": "जल पदचिह्न",
  "green_water": "हरा जल",
  "blue_water": "नीला जल",
  "grey_water": "ग्रे जल",
  "total_water": "कुल जल",
  "unit_label": "माप इकाई",
  "eco_tip_header": "💡 पर्यावरण टिप",
  "export_pdf": "PDF के रूप में निर्यात करें",
  "no_result": "उत्पाद नहीं मिला। कोई अलग नाम आज़माएं।",
  "history_header": "आपका जल पदचिह्न इतिहास",
  "history_empty": "अभी तक कोई स्कैन नहीं। उत्पाद स्कैन करना शुरू करें!",
  "language_label": "भाषा",
  "source_label": "डेटा स्रोत",
  "category_label": "श्रेणी",
  "confidence_label": "पहचान विश्वास"
}
```

---

## 🐍 Phase 2 — Build the Core Python Modules

### Step 2.1 — Create `app/footprint_engine.py`

```python
"""
footprint_engine.py
Handles all water footprint database lookups and fuzzy matching.
"""
import json
import os
from rapidfuzz import process, fuzz

# ── Paths ──────────────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_FOOTPRINT_DB_PATH = os.path.join(_DATA_DIR, "footprint_mapping.json")
_ECO_TIPS_PATH = os.path.join(_DATA_DIR, "eco_tips.json")

# ── Load DB once at import time ─────────────────────────────────────────────────
with open(_FOOTPRINT_DB_PATH, "r", encoding="utf-8") as f:
    FOOTPRINT_DB: dict = json.load(f)

with open(_ECO_TIPS_PATH, "r", encoding="utf-8") as f:
    ECO_TIPS_DB: dict = json.load(f)

_DB_KEYS = list(FOOTPRINT_DB.keys())


def lookup(product_label: str, threshold: int = 70) -> dict | None:
    """
    Look up water footprint data for a product.

    Args:
        product_label: Raw product name or YOLO label (snake_case or natural language).
        threshold: Minimum fuzzy match score (0–100). Default 70.

    Returns:
        dict with footprint data + matched key, or None if no match found.
    """
    # Normalize
    normalized = product_label.lower().replace(" ", "_").strip()

    # Exact match first
    if normalized in FOOTPRINT_DB:
        data = FOOTPRINT_DB[normalized].copy()
        data["matched_key"] = normalized
        data["match_score"] = 100
        return data

    # Fuzzy match
    result = process.extractOne(
        normalized,
        _DB_KEYS,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold
    )

    if result is None:
        return None

    matched_key, score, _ = result
    data = FOOTPRINT_DB[matched_key].copy()
    data["matched_key"] = matched_key
    data["match_score"] = score
    return data


def get_eco_tip(category: str) -> str:
    """Return a random eco tip for a product category."""
    import random
    tips = ECO_TIPS_DB.get(category, ECO_TIPS_DB["default"])
    return random.choice(tips)


def get_total_water(footprint_data: dict) -> int:
    """Sum green + blue + grey water."""
    return (
        footprint_data.get("green_water_litres", 0)
        + footprint_data.get("blue_water_litres", 0)
        + footprint_data.get("grey_water_litres", 0)
    )
```

### Step 2.2 — Create `app/translation_utils.py`

```python
"""
translation_utils.py
Handles loading and applying UI translations.
"""
import json
import os

_TRANSLATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "translations")

SUPPORTED_LANGUAGES = {
    "English": "en",
    "हिन्दी (Hindi)": "hi",
    "தமிழ் (Tamil)": "ta",
    "తెలుగు (Telugu)": "te",
    "বাংলা (Bengali)": "bn",
    "मराठी (Marathi)": "mr",
    "ಕನ್ನಡ (Kannada)": "kn",
}

_cache: dict[str, dict] = {}


def load_translation(lang_code: str) -> dict:
    """Load and cache a translation file by language code."""
    if lang_code in _cache:
        return _cache[lang_code]

    path = os.path.join(_TRANSLATIONS_DIR, f"{lang_code}.json")
    if not os.path.exists(path):
        # Fallback to English
        path = os.path.join(_TRANSLATIONS_DIR, "en.json")

    with open(path, "r", encoding="utf-8") as f:
        translations = json.load(f)

    _cache[lang_code] = translations
    return translations


def t(key: str, lang_code: str) -> str:
    """
    Translate a UI string key to the target language.

    Usage:
        t("search_button", "hi")  →  "खोजें"
    """
    translations = load_translation(lang_code)
    return translations.get(key, key)  # Fall back to key if missing
```

### Step 2.3 — Create `app/vision_module.py`

```python
"""
vision_module.py
Wraps YOLOv8 inference for image upload and webcam scan.
"""
import os
from PIL import Image
import numpy as np

_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model", "weights", "best.pt"
)
_FALLBACK_WEIGHTS = "yolov8n.pt"  # Use pretrained nano model if custom not found

_model = None


def _get_model():
    """Lazy-load the YOLO model (only once)."""
    global _model
    if _model is not None:
        return _model

    from ultralytics import YOLO

    weights = _WEIGHTS_PATH if os.path.exists(_WEIGHTS_PATH) else _FALLBACK_WEIGHTS
    _model = YOLO(weights)
    return _model


def detect_from_image(pil_image: Image.Image, conf_threshold: float = 0.4) -> list[dict]:
    """
    Run YOLOv8 inference on a PIL image.

    Returns:
        List of detections: [{"label": str, "confidence": float, "bbox": list}, ...]
        Sorted by confidence descending.
    """
    model = _get_model()
    img_array = np.array(pil_image)
    results = model(img_array, conf=conf_threshold, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            detections.append({
                "label": label,
                "confidence": round(confidence, 3),
                "bbox": bbox
            })

    # Sort by confidence, highest first
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def draw_boxes(pil_image: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw bounding boxes on a PIL image and return annotated image."""
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(pil_image.copy())
    img_copy = pil_image.copy()
    draw = ImageDraw.Draw(img_copy)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label_text = f"{det['label']} ({det['confidence']:.0%})"
        draw.rectangle([x1, y1, x2, y2], outline="#00BFFF", width=3)
        draw.text((x1 + 4, y1 + 4), label_text, fill="#FFFFFF")

    return img_copy
```

### Step 2.4 — Create `app/history_manager.py`

```python
"""
history_manager.py
Tracks per-session water footprint scan history using Streamlit session state.
"""
from datetime import datetime
import streamlit as st


def init_history():
    """Initialize history list in session state if not present."""
    if "scan_history" not in st.session_state:
        st.session_state.scan_history = []


def add_to_history(product: str, footprint_data: dict):
    """Add a scan result to session history."""
    init_history()
    st.session_state.scan_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "product": product,
        "total_litres": (
            footprint_data.get("green_water_litres", 0)
            + footprint_data.get("blue_water_litres", 0)
            + footprint_data.get("grey_water_litres", 0)
        ),
        "category": footprint_data.get("category", "unknown"),
    })


def get_history() -> list[dict]:
    """Return full scan history for this session."""
    init_history()
    return st.session_state.scan_history


def get_total_footprint() -> int:
    """Return cumulative total water footprint for this session."""
    return sum(item["total_litres"] for item in get_history())
```

---

## 🖥️ Phase 3 — Build the Streamlit Frontend

### Step 3.1 — Create `app/frontend.py`

This is the **main entry point**. It must include all four tabs.

```python
"""
frontend.py
AquaLens Streamlit Web Application — Main Entry Point

Run with:
    streamlit run app/frontend.py
"""

import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import sys, os

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from footprint_engine import lookup, get_eco_tip, get_total_water
from translation_utils import t, SUPPORTED_LANGUAGES
from vision_module import detect_from_image, draw_boxes
from history_manager import init_history, add_to_history, get_history, get_total_footprint

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaLens — Water Footprint Estimator",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar: Language Selector ─────────────────────────────────────────────────
with st.sidebar:
    st.image("app/assets/logo.png", use_column_width=True) if os.path.exists("app/assets/logo.png") else st.title("💧 AquaLens")
    st.markdown("---")
    lang_display = st.selectbox(
        "🌐 Language",
        list(SUPPORTED_LANGUAGES.keys()),
        index=0
    )
    lang_code = SUPPORTED_LANGUAGES[lang_display]
    st.markdown("---")
    st.markdown("**About AquaLens**")
    st.caption("An AI-powered app to reveal the hidden water cost of everyday products.")

# ── Title ──────────────────────────────────────────────────────────────────────
st.title(f"💧 {t('app_title', lang_code)}")
st.caption(t("app_tagline", lang_code))
st.markdown("---")

# ── Initialize history ─────────────────────────────────────────────────────────
init_history()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_search, tab_upload, tab_camera, tab_history = st.tabs([
    f"🔍 {t('tab_text_search', lang_code)}",
    f"🖼️ {t('tab_image_upload', lang_code)}",
    f"📷 {t('tab_camera_scan', lang_code)}",
    f"📊 {t('tab_history', lang_code)}",
])


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Render a footprint result card
# ─────────────────────────────────────────────────────────────────────────────
def render_footprint_card(product_name: str, data: dict, lang: str):
    """Display a formatted water footprint card for a product."""
    green = data.get("green_water_litres", 0)
    blue = data.get("blue_water_litres", 0)
    grey = data.get("grey_water_litres", 0)
    total = green + blue + grey
    unit = data.get("unit", "per_item").replace("_", " ")
    category = data.get("category", "default")
    source = data.get("source", "Unknown")

    display_name = data.get("matched_key", product_name).replace("_", " ").title()

    st.subheader(f"💧 {t('result_title', lang)}: **{display_name}**")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"🟢 {t('green_water', lang)}", f"{green:,} L")
    col2.metric(f"🔵 {t('blue_water', lang)}", f"{blue:,} L")
    col3.metric(f"⚫ {t('grey_water', lang)}", f"{grey:,} L")
    col4.metric(f"💧 {t('total_water', lang)}", f"{total:,} L")

    # Bar chart
    fig = go.Figure(go.Bar(
        x=[t("green_water", lang), t("blue_water", lang), t("grey_water", lang)],
        y=[green, blue, grey],
        marker_color=["#2ECC71", "#3498DB", "#95A5A6"],
        text=[f"{v:,} L" for v in [green, blue, grey]],
        textposition="auto"
    ))
    fig.update_layout(
        title=f"Water Footprint Breakdown ({unit})",
        yaxis_title="Litres",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metadata
    col_a, col_b = st.columns(2)
    col_a.info(f"📦 **{t('category_label', lang)}:** {category.title()}\n\n📏 **{t('unit_label', lang)}:** {unit}")
    col_b.info(f"📚 **{t('source_label', lang)}:** {source}")

    # Eco Tip
    tip = get_eco_tip(category)
    st.success(f"**{t('eco_tip_header', lang)}**\n\n{tip}")

    # Save to history
    add_to_history(display_name, data)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Text Search
# ─────────────────────────────────────────────────────────────────────────────
with tab_search:
    st.markdown("### 🔍 Search by Product Name")
    query = st.text_input(
        t("search_placeholder", lang_code),
        placeholder=t("search_placeholder", lang_code),
        key="text_search_input"
    )
    if st.button(t("search_button", lang_code), key="search_btn"):
        if query.strip():
            with st.spinner("Looking up water footprint..."):
                result = lookup(query.strip())
            if result:
                render_footprint_card(query, result, lang_code)
            else:
                st.warning(t("no_result", lang_code))
        else:
            st.warning("Please enter a product name.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Image Upload
# ─────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("### 🖼️ Upload a Product Image")
    uploaded_file = st.file_uploader(
        t("upload_button", lang_code),
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload"
    )
    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        with st.spinner("Detecting product..."):
            detections = detect_from_image(pil_image)

        if detections:
            annotated = draw_boxes(pil_image, detections)
            st.image(annotated, caption="Detection Result", use_column_width=True)

            top = detections[0]
            st.info(f"**Detected:** {top['label'].replace('_',' ').title()} — {t('confidence_label', lang_code)}: {top['confidence']:.0%}")

            result = lookup(top["label"])
            if result:
                render_footprint_card(top["label"], result, lang_code)
            else:
                st.warning(t("no_result", lang_code))
        else:
            st.warning("No product detected. Try a clearer image.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Camera Scan (placeholder — requires streamlit-webrtc setup)
# ─────────────────────────────────────────────────────────────────────────────
with tab_camera:
    st.markdown("### 📷 Live Camera Scan")
    st.info("📸 Live camera scanning uses your webcam to detect products in real time.")
    st.warning("⚙️ **Setup required:** This feature requires `streamlit-webrtc` and a TURN server for remote deployment. Works locally out of the box.")
    st.markdown("""
    **To enable camera scan:**
    1. Ensure `streamlit-webrtc` is installed: `pip install streamlit-webrtc`
    2. Run the app locally: `streamlit run app/frontend.py`
    3. Click the camera button in this tab
    """)
    # TODO: Implement real-time webrtc stream + frame inference
    # from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    # ... (implement VideoProcessor class with YOLO inference per frame)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: History Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:
    st.markdown(f"### 📊 {t('history_header', lang_code)}")
    history = get_history()

    if not history:
        st.info(t("history_empty", lang_code))
    else:
        total = get_total_footprint()
        st.metric("💧 Total Session Water Footprint", f"{total:,} Litres")

        import pandas as pd
        df = pd.DataFrame(history)
        st.dataframe(df, use_container_width=True)

        # Pie chart of usage by category
        cat_totals = df.groupby("category")["total_litres"].sum().reset_index()
        fig2 = go.Figure(go.Pie(
            labels=cat_totals["category"],
            values=cat_totals["total_litres"],
            hole=0.4
        ))
        fig2.update_layout(title="Water Footprint by Category (This Session)")
        st.plotly_chart(fig2, use_container_width=True)
```

---

## 🤖 Phase 4 — Build the YOLO Training Scripts

### Step 4.1 — Create `model/train_detector.py`

```python
"""
train_detector.py
Train a custom YOLOv8 model for product detection.

Usage:
    python model/train_detector.py

Requires:
    - data/yolo_dataset/ directory with images and labels
    - data/yolo_dataset/data.yaml config file
"""
import os
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_YAML = os.path.join("data", "yolo_dataset", "data.yaml")
BASE_MODEL = "yolov8s.pt"          # Start from YOLOv8-small pretrained weights
OUTPUT_DIR = os.path.join("model", "weights")
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16                    # Reduce to 8 if GPU memory < 6GB

def train():
    print(f"[AquaLens] Starting YOLOv8 training...")
    print(f"  Dataset config: {DATA_YAML}")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Epochs: {EPOCHS}")

    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"Dataset config not found: {DATA_YAML}\n"
            "Please prepare your dataset in data/yolo_dataset/ first."
        )

    model = YOLO(BASE_MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project="model",
        name="aqua_detector",
        save=True,
        patience=15,            # Early stopping
        augment=True,
        cos_lr=True,
    )

    # Copy best weights
    best = os.path.join("model", "aqua_detector", "weights", "best.pt")
    if os.path.exists(best):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        import shutil
        shutil.copy(best, os.path.join(OUTPUT_DIR, "best.pt"))
        print(f"[AquaLens] Best weights saved to {OUTPUT_DIR}/best.pt")

    return results

if __name__ == "__main__":
    train()
```

### Step 4.2 — Create `model/predict.py`

```python
"""
predict.py
Run inference on a single image for quick testing.

Usage:
    python model/predict.py --source path/to/image.jpg
    python model/predict.py --source 0            # webcam
"""
import argparse
from ultralytics import YOLO
import os

def predict(source: str, conf: float = 0.4):
    weights = os.path.join("model", "weights", "best.pt")
    if not os.path.exists(weights):
        print("Custom weights not found, using pretrained yolov8n.pt")
        weights = "yolov8n.pt"

    model = YOLO(weights)
    results = model(source, conf=conf, show=True)

    for r in results:
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            score = float(box.conf[0])
            print(f"  Detected: {label} ({score:.0%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Image path or webcam index (0)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    args = parser.parse_args()
    predict(args.source, args.conf)
```

---

## ✅ Phase 5 — Tests

### Step 5.1 — Create `tests/test_footprint_engine.py`

```python
"""Unit tests for footprint_engine.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from footprint_engine import lookup, get_total_water, get_eco_tip

def test_exact_match():
    result = lookup("cotton_shirt")
    assert result is not None
    assert result["matched_key"] == "cotton_shirt"
    assert result["match_score"] == 100

def test_fuzzy_match():
    result = lookup("cottn shirt")   # typo
    assert result is not None
    assert "cotton" in result["matched_key"]

def test_no_match():
    result = lookup("xyzquux123")
    assert result is None

def test_total_water():
    data = {"green_water_litres": 100, "blue_water_litres": 50, "grey_water_litres": 25}
    assert get_total_water(data) == 175

def test_eco_tip():
    tip = get_eco_tip("food")
    assert isinstance(tip, str)
    assert len(tip) > 0

def test_eco_tip_default():
    tip = get_eco_tip("unknown_category")
    assert isinstance(tip, str)
```

### Step 5.2 — Create `tests/test_translation_utils.py`

```python
"""Unit tests for translation_utils.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from translation_utils import t, load_translation, SUPPORTED_LANGUAGES

def test_english_translation():
    result = t("search_button", "en")
    assert result == "Search"

def test_hindi_translation():
    result = t("search_button", "hi")
    assert result == "खोजें"

def test_missing_key_fallback():
    result = t("nonexistent_key", "en")
    assert result == "nonexistent_key"

def test_missing_lang_fallback():
    # Unknown lang code should fall back to English
    result = t("search_button", "xx")
    assert result == "Search"

def test_all_languages_load():
    for lang_name, code in SUPPORTED_LANGUAGES.items():
        translations = load_translation(code)
        assert isinstance(translations, dict)
        assert "app_title" in translations
```

---

## 🚦 Phase 6 — Verification Checklist

After implementing all phases, verify the following:

### Functional Tests
```bash
# 1. Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# 2. Run unit tests
pytest tests/ -v --cov=app --cov-report=term-missing

# 3. Test footprint lookup directly
python -c "from app.footprint_engine import lookup; print(lookup('apple'))"

# 4. Start the app
streamlit run app/frontend.py
```

### Manual Verification Checklist
- [ ] Text search for "apple" returns water footprint card
- [ ] Text search for "cottn shrt" (typo) still finds cotton shirt (fuzzy)
- [ ] Text search for "xyzgarbage" shows "not found" message
- [ ] Image upload works and shows detection bounding boxes
- [ ] Language switcher changes all UI text (test Hindi)
- [ ] History tab shows all past scans in the session
- [ ] Eco tip changes each time (random)
- [ ] Plotly charts render correctly

---

## ⚠️ Common Pitfalls & Solutions

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: rapidfuzz` | Run `pip install rapidfuzz` |
| YOLO model download fails | Check internet connection; `yolov8n.pt` downloads from Ultralytics on first run |
| Streamlit webcam not working | Must run on localhost; remote requires TURN server |
| Translation file not found | Ensure `data/translations/en.json` exists as fallback |
| `footprint_mapping.json` KeyError | Use `data.get()` not `data[]` when accessing fields |
| GPU out of memory during training | Reduce `BATCH_SIZE` to 8 or 4 in `train_detector.py` |

---

## 📌 Decisions Already Made (Do NOT Change)

1. **Framework:** Streamlit (not Flask/Django) — simpler for AI demo apps
2. **Object Detection:** YOLOv8 — best balance of speed and accuracy for this use case
3. **Fuzzy Search:** `rapidfuzz` with `token_sort_ratio` — handles word order and typos
4. **Translation approach:** Static JSON files (not Google Translate API) — works offline
5. **Database format:** JSON (not SQLite) — sufficient for 100–500 products; no ORM needed
6. **Visualization:** Plotly (not Matplotlib) — interactive charts in Streamlit

---

*End of AGENT_GUIDE.md*
