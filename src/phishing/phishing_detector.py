# ============================================================================
# PHISHING EMAIL DETECTION - DASHBOARD INTEGRATION PACKAGE
# ============================================================================
# Author: Disha
# Project: CyberShield - Phishing Detection Module
# Date: March 2026
# ============================================================================

"""
QUICK SUMMARY FOR DASHBOARD TEAMMATE:
=====================================
1. MODEL FILES: 6 files in 'models/' folder (see below)
2. PREDICTION FUNCTION: predict_phishing(email_text) → returns dict
3. INPUT: Raw email text (subject + body combined)
4. OUTPUT: {'verdict': 'PHISHING'/'LEGITIMATE', 'confidence': 0.0-1.0}
"""

# ============================================================================
# SECTION 1: MODEL FILES NEEDED
# ============================================================================
"""
Download these from Google Drive: /phishing_detection_model/

REQUIRED FILES:
├── bert_lstm_phishing.pt      (423 MB) - Main BERT-LSTM model
├── xgb_model_v2.pkl           - XGBoost model
├── rf_model_v2.pkl            - Random Forest model
├── tfidf_v2.pkl               - TF-IDF vectorizer
├── scaler_v2.pkl              - Feature scaler
├── feature_columns_v2.pkl     - Feature column names
└── tokenizer/                 - BERT tokenizer folder
    ├── vocab.txt
    ├── tokenizer_config.json
    └── special_tokens_map.json

TOTAL SIZE: ~450 MB
"""

# ============================================================================
# SECTION 2: REQUIRED LIBRARIES (requirements.txt)
# ============================================================================
"""
torch>=1.9.0
transformers>=4.5.0
scikit-learn>=0.24.0
xgboost>=1.4.0
pandas>=1.2.0
numpy>=1.19.0
"""

# ============================================================================
# SECTION 3: THE PREDICTION FUNCTION
# ============================================================================

import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# ============== MODEL ARCHITECTURE (DO NOT MODIFY) ==============
class BERT_LSTM_Classifier(nn.Module):
    def __init__(self, bert_model, hidden_dim=128, num_layers=2, num_classes=2):
        super().__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(768, hidden_dim, num_layers, batch_first=True, 
                           bidirectional=True, dropout=0.3)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(bert_output.last_hidden_state)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return self.classifier(context_vector)


# ============== PHISHING DETECTOR CLASS ==============
class PhishingDetector:
    def __init__(self, model_path='models/'):
        """
        Initialize the phishing detector.
        
        Args:
            model_path: Path to folder containing all model files
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_loaded = False
        
        # These will be loaded when load_models() is called
        self.tokenizer = None
        self.bert_lstm_model = None
        self.xgb_model = None
        self.rf_model = None
        self.tfidf = None
        self.scaler = None
        self.feature_columns = None
    
    def load_models(self):
        """Load all model files. Call this once at startup."""
        print("Loading phishing detection models...")
        
        # Load BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(self.model_path, 'tokenizer/')
        )
        bert_base = BertModel.from_pretrained('bert-base-uncased')
        self.bert_lstm_model = BERT_LSTM_Classifier(bert_base).to(self.device)
        
        checkpoint = torch.load(
            os.path.join(self.model_path, 'bert_lstm_phishing.pt'),
            map_location=self.device
        )
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.bert_lstm_model.load_state_dict(state_dict, strict=False)
        self.bert_lstm_model.eval()
        print("  ✅ BERT-LSTM loaded")
        
        # Load XGBoost
        with open(os.path.join(self.model_path, 'xgb_model_v2.pkl'), 'rb') as f:
            self.xgb_model = pickle.load(f)
        print("  ✅ XGBoost loaded")
        
        # Load Random Forest
        with open(os.path.join(self.model_path, 'rf_model_v2.pkl'), 'rb') as f:
            self.rf_model = pickle.load(f)
        print("  ✅ Random Forest loaded")
        
        # Load TF-IDF, Scaler, Feature columns
        with open(os.path.join(self.model_path, 'tfidf_v2.pkl'), 'rb') as f:
            self.tfidf = pickle.load(f)
        with open(os.path.join(self.model_path, 'scaler_v2.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(self.model_path, 'feature_columns_v2.pkl'), 'rb') as f:
            self.feature_columns = pickle.load(f)
        print("  ✅ TF-IDF, Scaler loaded")
        
        self.models_loaded = True
        print("All models loaded successfully!")
    
    def _get_bert_score(self, text):
        """Get BERT-LSTM prediction score (0-100)"""
        try:
            encoding = self.tokenizer(
                str(text)[:5000],
                truncation=True,
                padding='max_length',
                max_length=256,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_lstm_model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                return probs[0][1].item() * 100
        except:
            return 50.0
    
    def _extract_features(self, text):
        """Extract manual features for XGBoost/RF"""
        text = str(text)
        text_lower = text.lower()
        
        return {
            'text_length': len(text),
            'word_count': len(text.split()),
            'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
            'url_count': len(re.findall(r'http[s]?://\S+', text)),
            'has_urgent': 1 if re.search(r'urgent|immediate|act now|limited time', text_lower) else 0,
            'has_verify': 1 if re.search(r'verify|confirm|validate|update', text_lower) else 0,
            'has_account': 1 if re.search(r'account|password|login|credential', text_lower) else 0,
            'has_money': 1 if re.search(r'\$|₹|money|payment|bank|credit|prize|won', text_lower) else 0,
            'has_threat': 1 if re.search(r'suspend|block|terminat|cancel|restrict', text_lower) else 0,
            'exclamation_count': text.count('!'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        }
    
    def _get_ml_score(self, text, model):
        """Get XGBoost or RF prediction score (0-100)"""
        try:
            features = self._extract_features(text)
            manual_df = pd.DataFrame([features])
            
            tfidf_features = self.tfidf.transform([str(text)[:5000]])
            tfidf_df = pd.DataFrame(
                tfidf_features.toarray(), 
                columns=self.tfidf.get_feature_names_out()
            )
            
            combined = pd.concat([manual_df.reset_index(drop=True), 
                                  tfidf_df.reset_index(drop=True)], axis=1)
            
            for col in self.feature_columns:
                if col not in combined.columns:
                    combined[col] = 0
            combined = combined[self.feature_columns]
            combined_scaled = self.scaler.transform(combined)
            
            return model.predict_proba(combined_scaled)[0][1] * 100
        except:
            return 50.0
    
    def _get_rule_score(self, text):
        """Rule-based detection with red/green flags"""
        text = str(text)
        text_lower = text.lower()
        red_flags = 0
        green_flags = 0
        
        # GREEN FLAGS (legitimate indicators)
        if any(e in text_lower for e in ['nptel', 'swayam', 'coursera', 'edx', 'udemy']):
            green_flags += 4
        if re.search(r'@.*\.(edu\.in|ac\.in|edu)', text_lower):
            green_flags += 3
        if any(d in text_lower for d in ['github.com', 'linkedin.com', 'google.com', 
                                          'microsoft.com', 'razorpay.com']):
            green_flags += 3
        if 'unsubscribe' in text_lower:
            green_flags += 1
        
        # RED FLAGS (phishing indicators)
        if re.search(r'(100%|guaranteed).{0,20}(placement|job)', text_lower):
            red_flags += 3
        if re.search(r'congratulat.{0,30}(shortlist|select|chosen|won)', text_lower):
            red_flags += 3
        if any(p in text_lower for p in ['linkedln', 'amaz0n', 'g00gle', 'sbi-', 'paytm-']):
            red_flags += 3
        if re.search(r'@.*(academy|teachnook|corizo)\.(in|com)', text_lower):
            red_flags += 2
        if re.search(r'account.{0,30}(block|suspend|terminat|locked)', text_lower):
            red_flags += 2
        if re.search(r'(fee|payment).{0,20}(required|₹|rs)', text_lower):
            red_flags += 3
        
        net_score = red_flags - green_flags
        
        if net_score >= 6: score = 95
        elif net_score >= 4: score = 85
        elif net_score >= 2: score = 65
        elif net_score >= 1: score = 55
        elif net_score <= -3: score = 15
        elif net_score <= -2: score = 25
        else: score = 40
        
        return {'score': score, 'red_flags': red_flags, 'green_flags': green_flags}
    
    def predict_phishing(self, email_text):
        """
        ================================================================
        MAIN PREDICTION FUNCTION - USE THIS IN DASHBOARD
        ================================================================
        
        Args:
            email_text (str): The email content (subject + body combined)
        
        Returns:
            dict: {
                'verdict': 'PHISHING' or 'LEGITIMATE',
                'confidence': float between 0.0 and 1.0,
                'score': float between 0 and 100,
                'details': {
                    'bert_score': float,
                    'xgb_score': float,
                    'rf_score': float,
                    'rule_score': float,
                    'red_flags': int,
                    'green_flags': int
                }
            }
        """
        if not self.models_loaded:
            raise Exception("Models not loaded! Call load_models() first.")
        
        # Get individual model scores
        bert_score = self._get_bert_score(email_text)
        xgb_score = self._get_ml_score(email_text, self.xgb_model)
        rf_score = self._get_ml_score(email_text, self.rf_model)
        rule_result = self._get_rule_score(email_text)
        rule_score = rule_result['score']
        
        # Dynamic weighting based on rule flags
        if rule_result['red_flags'] >= 5:
            weights = {'bert': 0.10, 'xgb': 0.20, 'rf': 0.15, 'rules': 0.55}
        elif rule_result['green_flags'] >= 3:
            weights = {'bert': 0.10, 'xgb': 0.15, 'rf': 0.10, 'rules': 0.65}
        else:
            weights = {'bert': 0.25, 'xgb': 0.30, 'rf': 0.20, 'rules': 0.25}
        
        # Calculate weighted score
        final_score = (
            bert_score * weights['bert'] +
            xgb_score * weights['xgb'] +
            rf_score * weights['rf'] +
            rule_score * weights['rules']
        )
        
        # Apply overrides for strong signals
        if rule_result['red_flags'] >= 5:
            final_score = max(final_score, 80)
        if rule_result['green_flags'] >= 4 and rule_result['red_flags'] == 0:
            final_score = min(final_score, 30)
        
        # Determine verdict
        verdict = 'PHISHING' if final_score >= 50 else 'LEGITIMATE'
        confidence = final_score / 100 if verdict == 'PHISHING' else (100 - final_score) / 100
        
        return {
            'verdict': verdict,
            'confidence': round(confidence, 3),
            'score': round(final_score, 2),
            'details': {
                'bert_score': round(bert_score, 2),
                'xgb_score': round(xgb_score, 2),
                'rf_score': round(rf_score, 2),
                'rule_score': round(rule_score, 2),
                'red_flags': rule_result['red_flags'],
                'green_flags': rule_result['green_flags']
            }
        }


# ============================================================================
# SECTION 4: SIMPLE WRAPPER FUNCTION (COPY THIS TO DASHBOARD)
# ============================================================================

# Global detector instance
_detector = None

def predict_phishing(email_text):
    """
    ================================================================
    SIMPLE FUNCTION FOR DASHBOARD INTEGRATION
    ================================================================
    
    Input:  email_text (str) - raw email content (subject + body)
    Output: dict with 'verdict' and 'confidence'
    
    Example:
        result = predict_phishing("Urgent! Verify your account now!")
        print(result)
        # {'verdict': 'PHISHING', 'confidence': 0.85}
    """
    global _detector
    
    if _detector is None:
        _detector = PhishingDetector(model_path='models/')
        _detector.load_models()
    
    result = _detector.predict_phishing(email_text)
    
    return {
        'verdict': result['verdict'],
        'confidence': result['confidence']
    }


# ============================================================================
# SECTION 5: USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Test the detector
    detector = PhishingDetector(model_path='models/')
    detector.load_models()
    
    # Test emails
    test_emails = [
        # Phishing examples
        "URGENT: Your account has been suspended! Click here to verify: http://amaz0n-verify.tk",
        "Congratulations! You've been selected for a $1000 prize. Claim now!",
        "Your PayPal account is limited. Update your information immediately.",
        
        # Legitimate examples
        "Hi team, the meeting has been rescheduled to 3 PM tomorrow.",
        "Your NPTEL course certificate is ready for download.",
        "GitHub: New sign-in to your account from Chrome on Windows."
    ]
    
    print("\n" + "="*70)
    print("PHISHING DETECTION TEST")
    print("="*70)
    
    for email in test_emails:
        result = detector.predict_phishing(email)
        print(f"\nEmail: {email[:60]}...")
        print(f"  → Verdict: {result['verdict']}")
        print(f"  → Confidence: {result['confidence']:.1%}")
        print(f"  → Score: {result['score']}")
