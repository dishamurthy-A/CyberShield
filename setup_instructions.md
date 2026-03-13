# Setup Instructions

## Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

## Installation Steps

### Step 1: Clone the repository
```bash
git clone https://github.com/dishamurthy-A/CyberShield.git
cd CyberShield
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download model files
Download models from Google Drive and place them in respective folders:
- Phishing models → `src/phishing/models/`
- Fake news models → `src/fake_news/models/`
- Deepfake models → `src/deepfakes/models/`

### Step 4: Run the dashboards

**Phishing Detection:**
```bash
cd src/phishing
streamlit run app.py
```

**Fake News Detection:**
```bash
cd src/fake_news
streamlit run main_app.py
```

**Deepfake Detection:**
```bash
cd src/deepfakes
streamlit run main_app.py
```

## Notes
- Models are not included in the repository due to size limits
- Download links are provided in README.md
```


---


Demo Video Link
===============

YouTube/Google Drive link will be added after recording.

[[PLACEHOLDER - https://youtu.be/H6l7mJxxDtY]]
