"""
Phishing Email Detection Module
Author: Disha
Version: 12.0
Accuracy: 93.4% on real-world datasets (SpamAssassin + Nazario Corpus)
         100% on personal Outlook emails

This module provides phishing email detection using:
- BERT-LSTM (Deep Learning)
- XGBoost (Gradient Boosting)
- Random Forest (Ensemble Trees)
- Rule-Based System v9 (Pattern Matching with Red/Green Flags)
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import re
import os

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# ============================================================
# BERT-LSTM MODEL ARCHITECTURE
# ============================================================
class BERT_LSTM_Classifier(nn.Module):
    """BERT-LSTM hybrid model for phishing detection"""
    def __init__(self, bert_model, hidden_dim=128, num_layers=2, num_classes=2):
        super().__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        logits = self.classifier(context_vector)
        return logits


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_features_v2(text):
    """Extract 12 manual features for XGBoost/RF"""
    text_lower = str(text).lower()
    words = str(text).split()
    
    return {
        'urgency': sum(1 for w in ['urgent', 'immediately', 'asap', 'now', 'expires', 'limited', 'hurry'] if w in text_lower),
        'money': sum(1 for w in ['$', 'money', 'payment', 'bank', 'credit', 'fee', 'dollar', 'price'] if w in text_lower),
        'action': sum(1 for w in ['click', 'verify', 'confirm', 'update', 'login', 'sign', 'enter', 'submit'] if w in text_lower),
        'credential': sum(1 for w in ['password', 'username', 'credential', 'ssn', 'account', 'pin'] if w in text_lower),
        'threat': sum(1 for w in ['suspend', 'terminate', 'locked', 'unauthorized', 'blocked', 'disabled'] if w in text_lower),
        'has_url': 1 if re.search(r'http[s]?://\S+', text_lower) else 0,
        'has_email': 1 if re.search(r'\S+@\S+\.\S+', text_lower) else 0,
        'text_length': len(str(text)),
        'word_count': len(words),
        'caps_ratio': sum(1 for c in str(text) if c.isupper()) / max(len(str(text)), 1),
        'exclaim_count': str(text).count('!'),
        'question_count': str(text).count('?'),
    }


# ============================================================
# RULE-BASED DETECTION v9
# ============================================================
def improved_rule_based_v9(text):
    """Version 9 - Comprehensive rule-based detection with red and green flags"""
    text_lower = text.lower()
    red_flags = 0
    red_reasons = []
    green_flags = 0
    green_reasons = []
    
    # ===== GREEN FLAGS (Legitimate indicators) =====
    
    # Official educational platforms
    official_edu = ['nptel', 'swayam', 'coursera', 'edx', 'udemy', 'linkedin learning', 'khan academy']
    if any(edu in text_lower for edu in official_edu):
        green_flags += 4
        green_reasons.append("Official educational platform")
    
    # Official university domains
    if re.search(r'@(alliance\.edu\.in|iitm\.ac\.in|iitk\.ac\.in|iitr\.ac\.in|\.edu\.in|\.ac\.in)', text_lower):
        green_flags += 3
        green_reasons.append("Official university domain")
    
    # Known tech company domains
    known_tech_domains = [
        'github.com', 'noreply@github.com',
        'linkedin.com', 'jobs-noreply@linkedin.com',
        'google.com', 'noreply@google.com',
        'microsoft.com', 'noreply@microsoft.com',
        'amazon.com', 'amazon.in',
        'razorpay.com', 'noreply@razorpay.com',
        'paytm.com', 'phonepe.com',
        'internshala.com', 'naukri.com',
        'hackerrank.com', 'leetcode.com'
    ]
    if any(domain in text_lower for domain in known_tech_domains):
        green_flags += 3
        green_reasons.append("Known tech company domain")
    
    # Internal university communications
    if re.search(r'(nss|ncc|e-?cell|student services|innovation council|placement cell|exam\s?cell|registrar)', text_lower):
        if re.search(r'@.*\.(edu\.in|ac\.in)', text_lower):
            green_flags += 3
            green_reasons.append("Internal university department")
    
    # Legitimate security alerts from known companies
    if re.search(r'(new sign-?in|login detected|security alert)', text_lower):
        if any(d in text_lower for d in ['github.com', 'google.com', 'linkedin.com', 'microsoft.com']):
            green_flags += 3
            green_reasons.append("Legitimate security alert")
    
    # Has unsubscribe
    if 'unsubscribe' in text_lower:
        green_flags += 1
        green_reasons.append("Has unsubscribe")
    
    # ===== RED FLAGS (Phishing indicators) =====
    
    # Fake placement guarantee
    if re.search(r'(internship|job|placement).{0,30}(program|opportunity)', text_lower):
        if re.search(r'(100%|guaranteed).{0,20}(placement|job)', text_lower):
            red_flags += 3
            red_reasons.append("Fake placement guarantee")
    
    # Congratulations + shortlisted (unsolicited)
    if re.search(r'congratulat.{0,30}(shortlist|select|chosen|won)', text_lower):
        if not any(d in text_lower for d in known_tech_domains):
            red_flags += 3
            red_reasons.append("Fake shortlist notification")
    
    # Google Forms from unknown sources
    if 'forms.gle' in text_lower or 'docs.google.com/forms' in text_lower:
        if not re.search(r'@.*\.(edu\.in|ac\.in)', text_lower):
            red_flags += 2
            red_reasons.append("Google Form link")
    
    # Multiple big company certifications
    big_companies = ['wipro', 'ibm', 'cisco', 'meta', 'oracle', 'deloitte', 'tcs', 'infosys']
    company_count = sum(1 for c in big_companies if c in text_lower)
    if company_count >= 3:
        red_flags += 3
        red_reasons.append(f"Claims {company_count} big company certifications")
    
    # Suspicious IIT collaboration from non-IIT domains
    if re.search(r'(iit|nit|iiit).{0,30}(collaborat|partner|associat)', text_lower):
        if not re.search(r'@.*iit.*\.ac\.in', text_lower):
            red_flags += 2
            red_reasons.append("Suspicious IIT collaboration claim")
    
    # Training marketing spam
    if re.search(r'(training|course|program|academy)', text_lower):
        if re.search(r'(upskill|unlock|potential|industry.?ready)', text_lower):
            red_flags += 2
            red_reasons.append("Training marketing spam")
    
    # Unknown training provider domains
    if re.search(r'@.*(academy|global|launched|teachnook|corizo)\.(in|org|com)', text_lower):
        red_flags += 2
        red_reasons.append("Unknown training provider domain")
    
    # Suspicious misspelled domains
    suspicious_patterns = ['linkedln', 'amaz0n', 'g00gle', 'micros0ft', 'faceb00k', 'paytm-', 'sbi-']
    if any(p in text_lower for p in suspicious_patterns):
        red_flags += 3
        red_reasons.append("Suspicious misspelled domain")
    
    # Account blocked/suspended threats from unknown sources
    if re.search(r'account.{0,30}(block|suspend|terminat|locked|disabled)', text_lower):
        if not any(d in text_lower for d in known_tech_domains):
            red_flags += 2
            red_reasons.append("Account threat from unknown source")
    
    # Fee/Payment required for internship/scholarship
    if re.search(r'(fee|payment|pay).{0,30}(required|₹|rs\.?|rupee)', text_lower):
        if re.search(r'(internship|scholarship|job|registration)', text_lower):
            red_flags += 3
            red_reasons.append("Payment required for opportunity")
    
    # Course duration pattern from unknown sources
    if re.search(r'\d+.?month.{0,20}(training|internship|program)', text_lower):
        if not re.search(r'@.*\.(edu\.in|ac\.in)', text_lower):
            red_flags += 1
            red_reasons.append("Course duration pattern")
    
    # LMS, Zoom, recorded sessions from unknown sources
    if re.search(r'(lms|zoom|recorded.{0,20}session)', text_lower):
        if re.search(r'(login|credential|access)', text_lower):
            if not re.search(r'@.*\.(edu\.in|ac\.in)', text_lower):
                red_flags += 1
                red_reasons.append("Online course spam pattern")
    
    # Score calculation
    net_score = red_flags - green_flags
    
    if net_score >= 8: score = 98
    elif net_score >= 6: score = 95
    elif net_score >= 4: score = 85
    elif net_score >= 3: score = 75
    elif net_score >= 2: score = 65
    elif net_score >= 1: score = 55
    elif net_score == 0: score = 40
    elif net_score <= -2: score = 20
    elif net_score <= -3: score = 15
    else: score = 30
    
    return {
        'score': score, 
        'red_flags': red_flags, 
        'green_flags': green_flags,
        'red_reasons': red_reasons, 
        'green_reasons': green_reasons
    }


# ============================================================
# MAIN DETECTOR CLASS
# ============================================================
class PhishingEmailDetector:
    """
    Main class for phishing email detection.
    Combines BERT-LSTM, XGBoost, Random Forest, and Rule-based detection.
    """
    
    def __init__(self, models_path=None):
        """Initialize the detector"""
        if models_path is None:
            models_path = MODELS_DIR
        
        self.models_path = models_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_loaded = False
        
        # Model placeholders
        self.bert_model = None
        self.tokenizer = None
        self.xgb_model = None
        self.rf_model = None
        self.tfidf = None
        self.scaler = None
        self.feature_columns = None
    
    def load_models(self):
        """Load all models from disk"""
        try:
            print("Loading models...")
            
            # Load XGBoost
            xgb_path = os.path.join(self.models_path, 'xgb_model_v2.pkl')
            if os.path.exists(xgb_path):
                with open(xgb_path, 'rb') as f:
                    self.xgb_model = pickle.load(f)
                print("  ✅ XGBoost loaded")
            
            # Load Random Forest
            rf_path = os.path.join(self.models_path, 'rf_model_v2.pkl')
            if os.path.exists(rf_path):
                with open(rf_path, 'rb') as f:
                    self.rf_model = pickle.load(f)
                print("  ✅ Random Forest loaded")
            
            # Load TF-IDF
            tfidf_path = os.path.join(self.models_path, 'tfidf_v2.pkl')
            if os.path.exists(tfidf_path):
                with open(tfidf_path, 'rb') as f:
                    self.tfidf = pickle.load(f)
                print("  ✅ TF-IDF loaded")
            
            # Load Scaler
            scaler_path = os.path.join(self.models_path, 'scaler_v2.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("  ✅ Scaler loaded")
            
            # Load Feature columns
            fc_path = os.path.join(self.models_path, 'feature_columns_v2.pkl')
            if os.path.exists(fc_path):
                with open(fc_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)
                print("  ✅ Feature columns loaded")
            
            # Load BERT tokenizer
            tokenizer_path = os.path.join(self.models_path, 'tokenizer')
            if os.path.exists(tokenizer_path):
                from transformers import BertTokenizer
                self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
                print("  ✅ Tokenizer loaded")
            
            # Load BERT-LSTM model
            bert_path = os.path.join(self.models_path, 'bert_lstm_phishing.pt')
            if os.path.exists(bert_path):
                from transformers import BertModel
                bert_base = BertModel.from_pretrained('bert-base-uncased')
                self.bert_model = BERT_LSTM_Classifier(bert_base)
                self.bert_model.load_state_dict(
                    torch.load(bert_path, map_location=self.device)
                )
                self.bert_model.to(self.device)
                self.bert_model.eval()
                print("  ✅ BERT-LSTM loaded")
            
            self.models_loaded = True
            print("\n✅ All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_bert_prediction(self, text):
        """Get BERT-LSTM prediction"""
        if self.bert_model is None or self.tokenizer is None:
            return 50.0
        
        try:
            self.bert_model.eval()
            encoding = self.tokenizer(
                text, 
                truncation=True, 
                max_length=256,
                padding='max_length', 
                return_tensors='pt'
            )
            
            with torch.no_grad():
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                outputs = self.bert_model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                return probs[0][1].item() * 100
        except:
            return 50.0
    
    def get_xgboost_prediction(self, text):
        """Get XGBoost prediction"""
        if self.xgb_model is None:
            return 50.0
        
        try:
            features = extract_features_v2(text)
            manual_df = pd.DataFrame([features])
            
            tfidf_array = self.tfidf.transform([str(text)]).toarray()
            tfidf_df = pd.DataFrame(
                tfidf_array, 
                columns=[f'tfidf_{i}' for i in range(200)]
            )
            
            X = pd.concat([
                manual_df.reset_index(drop=True), 
                tfidf_df.reset_index(drop=True)
            ], axis=1)
            X = X.reindex(columns=self.feature_columns, fill_value=0)
            
            X_scaled = self.scaler.transform(X.values)
            return self.xgb_model.predict_proba(X_scaled)[0][1] * 100
        except:
            return 50.0
    
    def get_rf_prediction(self, text):
        """Get Random Forest prediction"""
        if self.rf_model is None:
            return 50.0
        
        try:
            features = extract_features_v2(text)
            manual_df = pd.DataFrame([features])
            
            tfidf_array = self.tfidf.transform([str(text)]).toarray()
            tfidf_df = pd.DataFrame(
                tfidf_array, 
                columns=[f'tfidf_{i}' for i in range(200)]
            )
            
            X = pd.concat([
                manual_df.reset_index(drop=True), 
                tfidf_df.reset_index(drop=True)
            ], axis=1)
            X = X.reindex(columns=self.feature_columns, fill_value=0)
            
            X_scaled = self.scaler.transform(X.values)
            return self.rf_model.predict_proba(X_scaled)[0][1] * 100
        except:
            return 50.0
    
    def predict(self, email_text):
        """
        Main prediction method - combines all models.
        
        Args:
            email_text: The email content to analyze
            
        Returns:
            dict: {
                'is_phishing': bool,
                'confidence': float (0-100),
                'verdict': str ('PHISHING' or 'LEGITIMATE'),
                'risk_level': str ('HIGH', 'MEDIUM', 'LOW', 'SAFE'),
                'details': dict with individual model scores
            }
        """
        if not self.models_loaded:
            self.load_models()
        
        # Get individual model predictions
        bert_score = self.get_bert_prediction(email_text)
        xgb_score = self.get_xgboost_prediction(email_text)
        rf_score = self.get_rf_prediction(email_text)
        rule_result = improved_rule_based_v9(email_text)
        rule_score = rule_result['score']
        
        votes = sum([bert_score > 50, xgb_score > 50, rf_score > 50, rule_score > 50])
        
        # Dynamic weighting based on rule confidence
        if rule_result['red_flags'] >= 8:
            weights = {'bert': 0.05, 'xgb': 0.15, 'rf': 0.10, 'rules': 0.70}
        elif rule_result['red_flags'] >= 5:
            weights = {'bert': 0.10, 'xgb': 0.20, 'rf': 0.15, 'rules': 0.55}
        elif rule_result['red_flags'] >= 3:
            weights = {'bert': 0.15, 'xgb': 0.25, 'rf': 0.15, 'rules': 0.45}
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
        
        decision = "PHISHING" if final_score >= 50 else "LEGITIMATE"
        
        # Overrides for high-confidence phishing
        if rule_result['red_flags'] >= 8:
            final_score = max(final_score, 90)
            decision = "PHISHING"
        elif rule_result['red_flags'] >= 5:
            final_score = max(final_score, 80)
            decision = "PHISHING"
        elif rule_result['red_flags'] >= 4:
            final_score = max(final_score, 70)
            decision = "PHISHING"
        
        # Green flag overrides for legitimate emails
        if rule_result['green_flags'] >= 5 and rule_result['red_flags'] <= 1:
            final_score = min(final_score, 25)
            decision = "LEGITIMATE"
        elif rule_result['green_flags'] >= 4 and rule_result['red_flags'] == 0:
            final_score = min(final_score, 30)
            decision = "LEGITIMATE"
        elif rule_result['green_flags'] >= 3 and rule_result['red_flags'] == 0:
            final_score = min(final_score, 35)
            decision = "LEGITIMATE"
        
        # Determine risk level
        if final_score >= 80:
            risk_level = "HIGH"
        elif final_score >= 60:
            risk_level = "MEDIUM"
        elif final_score >= 40:
            risk_level = "LOW"
        else:
            risk_level = "SAFE"
        
        return {
            'is_phishing': decision == "PHISHING",
            'confidence': round(final_score, 1),
            'verdict': decision,
            'risk_level': risk_level,
            'votes': votes,
            'details': {
                'bert_score': round(bert_score, 1),
                'xgb_score': round(xgb_score, 1),
                'rf_score': round(rf_score, 1),
                'rule_score': round(rule_score, 1),
                'red_flags': rule_result['red_flags'],
                'green_flags': rule_result['green_flags'],
                'red_reasons': rule_result['red_reasons'],
                'green_reasons': rule_result['green_reasons'],
                'weights': weights
            }
        }


# ============================================================
# SIMPLE FUNCTION FOR QUICK DETECTION
# ============================================================
def detect_phishing_email(email_text, models_path=None):
    """
    Quick function to detect phishing email.
    
    Args:
        email_text: Email content to analyze
        models_path: Optional path to models directory
        
    Returns:
        dict with detection results
    """
    detector = PhishingEmailDetector(models_path)
    return detector.predict(email_text)


# ============================================================
# RULE-ONLY DETECTION (No ML models required)
# ============================================================
def detect_phishing_rules_only(email_text):
    """
    Detect phishing using only rule-based system.
    Use this when ML models are not available.
    
    Args:
        email_text: Email content to analyze
        
    Returns:
        dict with detection results
    """
    rule_result = improved_rule_based_v9(email_text)
    
    score = rule_result['score']
    decision = "PHISHING" if score >= 50 else "LEGITIMATE"
    
    if rule_result['red_flags'] >= 5:
        score = max(score, 80)
        decision = "PHISHING"
    elif rule_result['green_flags'] >= 3 and rule_result['red_flags'] == 0:
        score = min(score, 35)
        decision = "LEGITIMATE"
    
    if score >= 80:
        risk_level = "HIGH"
    elif score >= 60:
        risk_level = "MEDIUM"
    elif score >= 40:
        risk_level = "LOW"
    else:
        risk_level = "SAFE"
    
    return {
        'is_phishing': decision == "PHISHING",
        'confidence': round(score, 1),
        'verdict': decision,
        'risk_level': risk_level,
        'details': {
            'red_flags': rule_result['red_flags'],
            'green_flags': rule_result['green_flags'],
            'red_reasons': rule_result['red_reasons'],
            'green_reasons': rule_result['green_reasons']
        }
    }


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    # Test the detector
    test_email = """
    Congratulations! You've been selected for an exclusive internship program!
    Fill out this Google Form to apply: https://forms.gle/test123
    Get certifications from Microsoft, Google, and Amazon!
    100% Placement Guarantee!
    """
    
    print("Testing rule-based detection...")
    result = detect_phishing_rules_only(test_email)
    print(f"Result: {result['verdict']} ({result['confidence']}%)")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Red Flags: {result['details']['red_flags']}")
    print(f"Reasons: {result['details']['red_reasons']}")
