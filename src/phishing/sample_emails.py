# ============================================================================
# SAMPLE TEST EMAILS - 5 PHISHING + 5 LEGITIMATE
# ============================================================================
# Use these to test the phishing detection integration
# ============================================================================

SAMPLE_EMAILS = {
    # ==================== PHISHING EMAILS ====================
    "phishing_1": {
        "subject": "URGENT: Your Amazon Account Has Been Suspended",
        "body": """Dear Customer,

We have detected unusual activity on your Amazon account. Your account has been temporarily suspended.

To restore access, please verify your identity immediately by clicking the link below:

http://amaz0n-security-verify.tk/restore?id=8472

If you do not verify within 24 hours, your account will be permanently deleted.

Amazon Security Team""",
        "expected": "PHISHING"
    },
    
    "phishing_2": {
        "subject": "Congratulations! You've Been Shortlisted for TCS Hiring",
        "body": """Dear Candidate,

Congratulations! You have been shortlisted for the TCS NextStep hiring program.

To confirm your slot, please pay the registration fee of ₹2,999 using the link below:

https://tcs-hiring-registration.com/pay

This is a limited time offer. Seats are filling fast!

Regards,
TCS Recruitment Team
(This is NOT from official TCS)""",
        "expected": "PHISHING"
    },
    
    "phishing_3": {
        "subject": "Your SBI Account Will Be Blocked",
        "body": """ALERT: Your SBI account will be blocked within 24 hours!

Your KYC documents have expired. Update immediately to avoid account suspension.

Click here to update: http://sbi-kyc-update.in/verify

Enter your:
- Account Number
- ATM PIN
- OTP

SBI Customer Care
1800-XXX-XXXX""",
        "expected": "PHISHING"
    },
    
    "phishing_4": {
        "subject": "FREE Internship + 100% Placement Guarantee!",
        "body": """🎉 EXCLUSIVE OFFER 🎉

Join our 6-week internship program with GUARANTEED placement!

✅ Wipro Certification
✅ IBM Certification  
✅ Cisco Certification
✅ Meta Certification
✅ 100% Job Guarantee

Register NOW: https://forms.gle/fake123

Limited seats! Only ₹4,999 (90% scholarship applied)

From: career@teachnook-academy.in""",
        "expected": "PHISHING"
    },
    
    "phishing_5": {
        "subject": "LinkedIn: Verify Your Account Immediately",
        "body": """LinkedIn Security Alert

We noticed a suspicious login to your LinkedIn account.

Your account has been temporarily limited. To restore full access, verify your identity:

http://linkedln-verify.com/security

If this was you, please verify within 12 hours.
If this wasn't you, your account may be compromised.

LinkedIn Security Team""",
        "expected": "PHISHING"
    },
    
    # ==================== LEGITIMATE EMAILS ====================
    "legitimate_1": {
        "subject": "GitHub: New sign-in to your account",
        "body": """Hi there,

We noticed a new sign-in to your GitHub account.

Device: Chrome on Windows
Location: Bengaluru, India
Time: March 12, 2026 at 10:30 AM IST

If this was you, no further action is needed.
If this wasn't you, please secure your account: https://github.com/settings/security

Thanks,
The GitHub Team

To unsubscribe from these emails, visit your notification settings.""",
        "expected": "LEGITIMATE"
    },
    
    "legitimate_2": {
        "subject": "NPTEL: Your Course Certificate is Ready",
        "body": """Dear Learner,

Congratulations on successfully completing the NPTEL course:
"Introduction to Machine Learning" (12 weeks)

Your certificate is now available for download.

Download Certificate: https://nptel.ac.in/certificates

Course Score: 78%
Certificate Type: Elite + Silver

Best regards,
NPTEL Team
IIT Madras

This is an automated email from nptel.iitm.ac.in""",
        "expected": "LEGITIMATE"
    },
    
    "legitimate_3": {
        "subject": "Team Meeting Rescheduled to 3 PM",
        "body": """Hi Team,

Just a quick update - tomorrow's project sync meeting has been moved from 2 PM to 3 PM.

Same Zoom link as before.

Agenda:
1. Sprint review
2. Bug fixes update
3. Next week planning

Please update your calendars.

Thanks,
Priya
Project Manager""",
        "expected": "LEGITIMATE"
    },
    
    "legitimate_4": {
        "subject": "Internshala: 3 New Internships Match Your Profile",
        "body": """Hi Disha,

Based on your preferences, we found 3 new internships for you:

1. Data Science Intern at Flipkart (Remote)
   Stipend: ₹25,000/month
   
2. ML Engineer Intern at Swiggy (Bangalore)
   Stipend: ₹30,000/month
   
3. Research Intern at Microsoft (Hyderabad)
   Stipend: ₹50,000/month

Apply now: https://internshala.com/applications

Good luck!
Team Internshala

Unsubscribe: https://internshala.com/unsubscribe""",
        "expected": "LEGITIMATE"
    },
    
    "legitimate_5": {
        "subject": "Razorpay: Payment Received - ₹1,299",
        "body": """Payment Successful ✓

Hi Disha,

Your payment of ₹1,299 to Zomato has been processed.

Transaction ID: rzp_8472947294
Date: March 12, 2026
Method: UPI (Google Pay)

View receipt: https://razorpay.com/receipt/rzp_8472947294

If you didn't make this payment, contact us immediately.

Thanks,
Razorpay""",
        "expected": "LEGITIMATE"
    }
}


# ============================================================================
# HOW TO USE THESE TEST EMAILS
# ============================================================================
"""
from sample_emails import SAMPLE_EMAILS
from phishing_detector import predict_phishing

for name, email_data in SAMPLE_EMAILS.items():
    email_text = email_data['subject'] + ' ' + email_data['body']
    result = predict_phishing(email_text)
    
    correct = "✅" if result['verdict'] == email_data['expected'] else "❌"
    print(f"{correct} {name}: {result['verdict']} ({result['confidence']:.1%})")
"""
