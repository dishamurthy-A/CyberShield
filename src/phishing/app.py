"""
🛡️ Phishing Email Detection Dashboard
Author: Disha
Version: 12.0
Accuracy: 93.4% on real-world datasets

This Streamlit app provides a user-friendly interface for detecting phishing emails
using a combination of BERT-LSTM, XGBoost, Random Forest, and Rule-based detection.
"""

import streamlit as st
import sys
import os

# Add modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.email_detector import (
    PhishingEmailDetector, 
    detect_phishing_rules_only,
    improved_rule_based_v9
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .phishing-box {
        background-color: #FEE2E2;
        border: 2px solid #DC2626;
    }
    .safe-box {
        background-color: #D1FAE5;
        border: 2px solid #059669;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .flag-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
    }
    .red-flag {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .green-flag {
        background-color: #D1FAE5;
        color: #065F46;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/security-checked.png", width=80)
    st.title("🛡️ Phishing Detector")
    st.markdown("---")
    
    st.markdown("### 📊 Model Information")
    st.markdown("""
    **Models Used:**
    - 🧠 BERT-LSTM (99.28%)
    - 🚀 XGBoost (94.17%)
    - 🌲 Random Forest (92.05%)
    - 📋 Rule-Based v9
    
    **Real-World Accuracy:**
    - SpamAssassin: 91.5%
    - Nazario 2023: 93.0%
    - Nazario 2024: 96.0%
    - Nazario 2025: 93.0%
    - **Overall: 93.4%**
    """)
    
    st.markdown("---")
    
    # Mode selection
    detection_mode = st.radio(
        "Detection Mode:",
        ["🚀 Full (All Models)", "⚡ Quick (Rules Only)"],
        help="Full mode uses all ML models. Quick mode uses only rule-based detection."
    )
    
    st.markdown("---")
    st.markdown("### 👩‍💻 About")
    st.markdown("""
    Created by **Disha**
    
    Part of CyberShield Project
    """)

# ============================================================
# MAIN CONTENT
# ============================================================
st.markdown('<p class="main-header">🛡️ Phishing Email Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by BERT-LSTM + XGBoost + Random Forest + Rule-Based Analysis</p>', unsafe_allow_html=True)

# ============================================================
# EMAIL INPUT
# ============================================================
st.markdown("### 📧 Enter Email Content")

email_text = st.text_area(
    "Paste the email content here (including subject, sender, and body):",
    height=250,
    placeholder="""Example:
From: hr@company-careers.com
Subject: Congratulations! You've been selected!

Dear Student,

You have been selected for our exclusive internship program...
""",
    key="email_input"
)

# Sample emails for testing
with st.expander("📝 Load Sample Email"):
    sample_type = st.selectbox(
        "Select sample type:",
        ["Phishing - Internship Scam", "Phishing - Bank Scam", "Legitimate - University Notice", "Legitimate - GitHub Alert"]
    )
    
    samples = {
        "Phishing - Internship Scam": """
From: hr@launchedglobal.in
Subject: Congratulations! You've been shortlisted for internship!

Dear Student,

Congratulations on being shortlisted for the internship program!

LaunchEd Global in Collaboration with IIT KHARAGPUR is organizing a 2-month training & internship program with certifications from WIPRO Dice ID, IBM, CISCO, META, and AICTE-approved.

Fill out the Google Form today: https://forms.gle/example123

What You'll Receive:
* Certificate from IIT-KHARAGPUR
* Verified by WIPRO Dice ID
* 100% Placement Assistance

Thanks & Regards,
HR Team
        """,
        "Phishing - Bank Scam": """
From: alert@sbi-customercare.in
Subject: URGENT: Your SBI Account is Blocked

Dear Valued Customer,

Your SBI account has been temporarily blocked due to incomplete KYC verification.

To unblock your account immediately:
1. Click here: https://sbi-kyc-update.com/verify
2. Enter your account details
3. Submit OTP received on your mobile

Failure to verify within 24 hours will result in permanent account closure.

Regards,
SBI Customer Care
        """,
        "Legitimate - University Notice": """
From: examcell@alliance.edu.in
Subject: End Semester Examination Schedule - April 2026

Dear Students,

The End Semester Examinations for April 2026 have been scheduled.

Exam Dates: April 15 - April 30, 2026
Reporting Time: 9:00 AM
Venue: Main Examination Hall, Block A

Important Instructions:
1. Carry your University ID Card
2. Reach 30 minutes before exam time
3. Download hall ticket from student portal

For queries, contact Exam Cell at examcell@alliance.edu.in

Regards,
Controller of Examinations
Alliance University
        """,
        "Legitimate - GitHub Alert": """
From: noreply@github.com
Subject: [GitHub] A new sign-in to your account

Hey user!

A new sign-in was detected on your GitHub account.

Device: Chrome on Windows
Location: Bangalore, India
Time: March 9, 2026 at 10:30 AM IST

If this was you, you can ignore this message.

If this wasn't you, please secure your account immediately:
https://github.com/settings/security

Thanks,
The GitHub Team
        """
    }
    
    if st.button("Load Sample"):
        st.session_state.email_input = samples[sample_type]
        st.rerun()

# ============================================================
# ANALYZE BUTTON
# ============================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button(
        "🔍 Analyze Email", 
        type="primary", 
        use_container_width=True,
        disabled=len(email_text.strip()) < 10
    )

# ============================================================
# RESULTS
# ============================================================
if analyze_button and email_text.strip():
    with st.spinner("🔄 Analyzing email..."):
        
        # Detect based on mode
        if "Full" in detection_mode:
            # Try to load full detector
            try:
                detector = PhishingEmailDetector()
                if detector.load_models():
                    result = detector.predict(email_text)
                else:
                    st.warning("⚠️ ML models not found. Using rule-based detection.")
                    result = detect_phishing_rules_only(email_text)
            except Exception as e:
                st.warning(f"⚠️ Could not load ML models: {e}. Using rule-based detection.")
                result = detect_phishing_rules_only(email_text)
        else:
            result = detect_phishing_rules_only(email_text)
    
    st.markdown("---")
    st.markdown("### 📊 Analysis Results")
    
    # Main result display
    if result['is_phishing']:
        st.markdown(f"""
        <div class="result-box phishing-box">
            <h2 style="color: #DC2626; margin: 0;">⚠️ PHISHING DETECTED</h2>
            <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                Confidence: <strong>{result['confidence']}%</strong> | 
                Risk Level: <strong>{result['risk_level']}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box safe-box">
            <h2 style="color: #059669; margin: 0;">✅ EMAIL APPEARS SAFE</h2>
            <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                Confidence: <strong>{100 - result['confidence']:.1f}%</strong> | 
                Risk Level: <strong>{result['risk_level']}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed metrics
    st.markdown("### 🔬 Detailed Analysis")
    
    details = result['details']
    
    # Model scores (if available)
    if 'bert_score' in details:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🧠 BERT-LSTM", 
                f"{details['bert_score']}%",
                delta="Phishing" if details['bert_score'] > 50 else "Safe"
            )
        
        with col2:
            st.metric(
                "🚀 XGBoost", 
                f"{details['xgb_score']}%",
                delta="Phishing" if details['xgb_score'] > 50 else "Safe"
            )
        
        with col3:
            st.metric(
                "🌲 Random Forest", 
                f"{details['rf_score']}%",
                delta="Phishing" if details['rf_score'] > 50 else "Safe"
            )
        
        with col4:
            st.metric(
                "📋 Rules", 
                f"{details['rule_score']}%",
                delta="Phishing" if details['rule_score'] > 50 else "Safe"
            )
    
    # Flags
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚩 Red Flags Detected")
        if details['red_flags'] > 0:
            for reason in details['red_reasons']:
                st.markdown(f"""
                <div class="flag-item red-flag">
                    ⚠️ {reason}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No red flags detected")
    
    with col2:
        st.markdown("#### ✅ Green Flags Detected")
        if details['green_flags'] > 0:
            for reason in details['green_reasons']:
                st.markdown(f"""
                <div class="flag-item green-flag">
                    ✓ {reason}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No green flags detected")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    if result['risk_level'] == "HIGH":
        st.error("""
        **⚠️ HIGH RISK - Do NOT interact with this email!**
        - Do not click any links
        - Do not download attachments
        - Do not reply with personal information
        - Report to your IT department
        - Mark as spam/phishing in your email client
        """)
    elif result['risk_level'] == "MEDIUM":
        st.warning("""
        **⚠️ MEDIUM RISK - Proceed with caution**
        - Verify the sender's identity through official channels
        - Do not click links directly - type the URL manually
        - Contact the supposed sender through known contact info
        - When in doubt, delete the email
        """)
    elif result['risk_level'] == "LOW":
        st.info("""
        **ℹ️ LOW RISK - Some suspicious elements detected**
        - Review carefully before taking action
        - Verify if you expected this email
        - Check sender's email address carefully
        """)
    else:
        st.success("""
        **✅ SAFE - This email appears legitimate**
        - Normal business/personal communication detected
        - No significant phishing indicators found
        - Still exercise general email safety practices
        """)

elif analyze_button:
    st.warning("Please enter email content to analyze.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 1rem;'>
    <p>🛡️ <strong>Phishing Email Detection System v12.0</strong></p>
    <p>Trained on 82,000+ emails | Tested on SpamAssassin & Nazario Corpus</p>
    <p>Created by Disha | CyberShield Project</p>
</div>
""", unsafe_allow_html=True)
