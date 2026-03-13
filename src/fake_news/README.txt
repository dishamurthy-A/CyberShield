================================================================
  CYBERSHIELD — AI-Powered Cyber Threat Detection System
  IEEE Research Project | Alliance University, Bengaluru
================================================================

OVERVIEW
--------
CyberShield is an integrated AI dashboard that detects three types
of cyber threats in real time:

  1. Deepfake Detection  — EfficientNet-B0 (image + video)
  2. Phishing Detection  — BERT-LSTM + XGBoost + Random Forest
  3. Fake News Detection — DistilBERT + TF-IDF Voting Ensemble


----------------------------------------------------------------
FOLDER STRUCTURE
----------------------------------------------------------------

CyberShield/
  main_app.py                  <- Main Streamlit dashboard
  requirements.txt             <- Python dependencies
  README.txt                   <- This file

  fakenews/
    fakenews_detector.py       <- Fake news detection module
    tfidf_vectorizer_final.pkl <- TF-IDF model (trained)
    lr_model_final.pkl         <- Logistic Regression model
    distilbert_model_final/    <- DistilBERT fine-tuned model
      config.json
      model.safetensors
      tokenizer.json
      tokenizer_config.json

  modules/
    email_detector.py          <- Phishing detection module (Disha)

  models/                      <- Phishing model files (Disha)
    bert_lstm_phishing.pt
    xgb_model_v2.pkl
    rf_model_v2.pkl
    tfidf_v2.pkl
    scaler_v2.pkl
    feature_columns_v2.pkl
    tokenizer/

  NOTE: .pth model files for deepfake detection are stored locally
  and not uploaded to GitHub due to large file size (use Git LFS
  or store in Google Drive and update the path in main_app.py).


----------------------------------------------------------------
INSTALLATION
----------------------------------------------------------------

Step 1 — Clone the repository
  git clone https://github.com/yourusername/CyberShield.git
  cd CyberShield

Step 2 — Create a virtual environment (recommended)
  python -m venv venv
  venv\Scripts\activate          (Windows)
  source venv/bin/activate       (Mac/Linux)

Step 3 — Install dependencies
  pip install -r requirements.txt

Step 4 — Add your model files
  - Copy your .pth deepfake model files to the Downloads folder
    or update the DOWNLOADS path in main_app.py
  - Copy fakenews model files to the fakenews/ folder
  - Copy phishing model files to the models/ folder


----------------------------------------------------------------
RUNNING THE DASHBOARD
----------------------------------------------------------------

  cd CyberShield
  py -m streamlit run main_app.py       (Windows)
  streamlit run main_app.py             (Mac/Linux)

Then open: http://localhost:8501


----------------------------------------------------------------
UPDATING MODEL PATHS
----------------------------------------------------------------

Open main_app.py and update these two lines near the top:

  DOWNLOADS = Path(r'C:\Users\UMME\Downloads')

  Change to your own path where the .pth model files are stored.
  For example:
  DOWNLOADS = Path(r'C:\Users\YourName\Downloads')


----------------------------------------------------------------
MODULE DETAILS
----------------------------------------------------------------

1. FAKE NEWS DETECTION
   - Primary Model  : DistilBERT (distilbert-base-uncased)
   - Secondary Model: TF-IDF Voting Ensemble (LR + SGD + SVC)
   - Training Data  : 45,000 articles (ISOT + FakeReal + WELFake)
   - Accuracy       : DistilBERT 98% | TF-IDF+LR 96.5%
   - Ensemble       : 70% DistilBERT + 30% TF-IDF
   - Author         : Sahana

2. DEEPFAKE DETECTION
   - Model          : EfficientNet-B0 (4 variants)
   - Supports       : Images (JPG/PNG) and Videos (MP4/AVI/MOV)
   - Accuracy       : 93.5% on FaceForensics++
   - Features       : GradCAM heatmap, face detection, adversarial defense
   - Author         : Umme

3. PHISHING EMAIL DETECTION
   - Models         : BERT-LSTM + XGBoost v2 + Random Forest v2 + Rules
   - Accuracy       : BERT-LSTM 99.28% | XGBoost 94.17%
   - Features       : Red/green flags, risk level, recommendation
   - Author         : Disha


----------------------------------------------------------------
PERFORMANCE SUMMARY
----------------------------------------------------------------

  Module           Model            Accuracy   F1-Score
  -------------------------------------------------------
  Fake News        DistilBERT       98.0%      99%
  Fake News        TF-IDF + LR      96.5%      97%
  Deepfake         EfficientNet-B0  93.5%      93.19%
  Phishing         BERT-LSTM        99.28%     99.24%
  Phishing         XGBoost v2       94.17%     93.10%


----------------------------------------------------------------
TECHNOLOGY STACK
----------------------------------------------------------------

  Frontend    : Streamlit, Custom CSS (Rajdhani + Share Tech Mono)
  Deep Learning: PyTorch, HuggingFace Transformers, timm
  ML Models   : Scikit-learn (TF-IDF, LR, SGD, SVC), XGBoost
  Vision      : OpenCV, Pillow, torchvision
  Plotting    : Matplotlib
  Other       : Scipy, NumPy, Pandas


----------------------------------------------------------------
NOTES FOR GITHUB
----------------------------------------------------------------

  - Do NOT upload large .pth / .pkl / .pt / .safetensors files
    directly to GitHub (they exceed the 100MB limit).
  - Use Git LFS or store models in Google Drive and share link.
  - Add a .gitignore file to exclude model files (see below).

  Recommended .gitignore entries:
    *.pth
    *.pt
    *.pkl
    *.safetensors
    __pycache__/
    .streamlit/
    venv/
    *.pyc


----------------------------------------------------------------
REFERENCES
----------------------------------------------------------------

  [1] Oad et al. 2024 — Enhanced BERT for Fake News — IEEE Access
  [2] Verma et al. 2021 — WELFake Dataset — IEEE Trans. CSS
  [3] Sanh et al. 2020 — DistilBERT — arXiv:1910.01108
  [4] Rossler et al. 2019 — FaceForensics++ — ICCV 2019


================================================================
  Alliance School of Advanced Computing
  Faculty of Engineering and Technology
  Alliance University, Bengaluru, Karnataka 562106
  www.alliance.edu.in
================================================================
