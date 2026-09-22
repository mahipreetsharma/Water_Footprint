# 💧 AquaLens: AI-Powered Water Footprint Estimator

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)

## 📌 Problem Statement
The water footprint measures the amount of water used to produce each of the goods and services we use. Preventing severe drought in water-stressed areas requires efficient water usage and readily available data. 

**AquaLens** tackles this by using digital technologies (AI and Computer Vision) to create a user-friendly app. By feeding inputs or simply scanning items through a camera (like Google Lens), users can instantly see the water footprints of daily items. The app includes local language support to ensure pan-India usage and sensitize citizens about their hidden water consumption.

---

## ✨ Features
*   **Object Detection:** Real-time scanning of daily items (clothing, food, electronics) using a custom-trained YOLOv8 model.
*   **Water Footprint Database:** Detailed breakdown of Green (rainwater), Blue (surface/groundwater), and Grey (polluted) water footprints based on Water Footprint Network data.
*   **Multilingual Support:** Pan-India accessibility with UI and data translations into local languages.
*   **Web-Based Interface:** Lightweight, cross-platform frontend built entirely in Python using Streamlit.

---

## 🏗️ Project Architecture

```text
📁 AquaLens-Project/
├── 📁 data/
│   ├── 📄 footprint_mapping.json  # Database mapping object classes to footprint values
│   ├── 📁 raw_images/             # Unprocessed training images (.jpg, .png)
│   └── 📁 labels/                 # YOLO format bounding box annotations (.txt)
├── 📁 model/
│   ├── 📄 train_detector.py       # Python script for training the YOLO model
│   ├── 📄 predict.py              # Inference script for testing predictions
│   └── 📄 best_weights.pt         # Saved PyTorch model weights after training
├── 📁 app/
│   ├── 📄 frontend.py             # Streamlit web application (UI)
│   ├── 📄 translation_utils.py    # Local language support/translation mapping
│   └── 📁 assets/                 # UI assets like icons and logos
├── 📄 requirements.txt            # Project dependencies and versions
└── 📄 README.md                   # Project documentation
```

---

## 📊 Data Used

1.  **Computer Vision Dataset (Images):** 
    *   **Format:** `.jpg` and `.png` image files with corresponding `.txt` YOLO bounding box annotations.
    *   **Usage:** Used to train the YOLOv8 model to recognize everyday items (e.g., `cotton_shirt`, `apple`, `coffee_cup`).
2.  **Water Footprint Database (Tabular):**
    *   **Format:** `.json` file (`footprint_mapping.json`).
    *   **Usage:** Acts as a local database mapping the detected object classes to their respective water footprint metrics (in liters).

---

## 🚀 Step-by-Step Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/AquaLens.git
cd AquaLens
```

### 2. Set Up Virtual Environment
It is recommended to use a virtual environment to manage dependencies.
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies include `ultralytics`, `streamlit`, `opencv-python`, `pandas`)*

### 4. Train the Model (Optional)
If you wish to retrain the object detection model with new data:
```bash
python model/train_detector.py
```
*Note: This requires your dataset to be properly formatted and linked in a `data.yaml` file.*

### 5. Run the Application
Launch the Streamlit frontend. This will open the application in your default web browser.
```bash
streamlit run app/frontend.py
```

---

## 🛠️ Tech Stack
*   **Machine Learning:** PyTorch, Ultralytics YOLOv8
*   **Frontend/Backend:** Python, Streamlit
*   **Data Processing:** OpenCV, Pandas, JSON

---

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your suggested improvements, bug fixes, or dataset expansions.