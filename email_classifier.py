import pandas as pd
import random
import os
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
 
# optional SMOTE for handling class imbalance
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
 
# ─────────────────────────────────────────────
#  EXPANDED KEYWORD BANK FOR AUTO-LABELLING
# ─────────────────────────────────────────────
def auto_label(subject, body):
    text = f"{subject} {body}".lower()
 
    if any(word in text for word in [
        "security", "login", "sign-in", "unauthorized", "unusual activity",
        "alert", "account", "password", "reset password", "new device", "breach",
        "suspicious", "verification", "recovery", "2fa", "authentication",
        "access denied", "phishing", "blocked access", "attempted login",
        "unrecognized device", "verify identity", "compromised", "secure link",
        "session expired", "malware", "fraud", "identity theft", "data breach",
        "cyber", "otp", "one time password", "intrusion", "firewall", "vpn"
    ]):
        return "security"
 
    elif any(word in text for word in [
        "urgent", "asap", "immediate", "important", "final reminder", "last date",
        "action needed", "due today", "deadline", "today itself", "respond now",
        "approval required", "pending approval", "critical", "high priority",
        "emergency", "respond by", "submit", "submission deadline", "non-compliance",
        "overdue", "past due", "time-sensitive", "immediately", "escalation",
        "must act", "requires attention", "time critical"
    ]):
        return "urgent"
 
    elif any(word in text for word in [
        "meeting", "zoom", "google meet", "webex", "teams", "discussion",
        "conference", "calendar invite", "faculty meeting", "board meeting",
        "virtual session", "meeting scheduled", "meeting request", "call with",
        "review meeting", "panel discussion", "committee meeting", "strategic meeting",
        "agenda", "minutes", "attendees", "rsvp", "sync", "standup",
        "one-on-one", "all hands", "1:1"
    ]):
        return "meeting"
 
    elif any(word in text for word in [
        "schedule", "timetable", "revised schedule", "slot", "slot change",
        "class timing", "rescheduling", "updated slot", "adjusted slot",
        "calendar update", "exam duty", "invigilation", "lecture timing",
        "exam schedule", "exam plan", "lecture rescheduled", "teaching slot",
        "revised calendar", "updated roster", "schedule note", "duty schedule",
        "appointment", "booking", "reservation", "time slot", "availability",
        "office hours", "session time"
    ]):
        return "schedule"
 
    elif any(word in text for word in [
        "circular", "notice", "announcement", "guideline", "update", "notification",
        "rules", "regulation", "policy", "protocol", "minutes of meeting",
        "documentation", "instructions", "procedures", "report", "summary",
        "handbook", "exam policy", "grading policy", "evaluation method",
        "admin note", "official announcement", "institutional update", "brief",
        "memo", "new process", "attachment", "circular released", "newsletter",
        "press release", "bulletin", "release notes", "digest"
    ]):
        return "information"
 
    elif any(word in text for word in [
        "birthday", "invitation", "celebration", "reunion", "dinner", "lunch",
        "family", "outing", "get together", "thank you", "congratulations",
        "best wishes", "personal", "call me", "catch up", "coffee", "greetings",
        "fun", "festival", "holiday", "gift", "wedding", "marriage",
        "hope you're doing well", "miss you", "free this evening",
        "event pictures", "how are you", "checking in", "weekend", "vacation",
        "trip", "party"
    ]):
        return "personal"
 
    return "information"
 
 
# ─────────────────────────────────────────────
#  LARGER SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────
def generate_synthetic_emails(n_per_class=80):
    templates = {
        "meeting": {
            "subjects": [
                "Department Meeting at 10 AM", "Urgent: Research Review Meeting",
                "Schedule Confirmation for Zoom Call", "Board Meeting Next Monday",
                "Team Sync Invite", "Weekly Standup — Please Join",
                "Agenda for Tomorrow's Faculty Meeting", "One-on-One with Dean",
                "Project Review Session Scheduled", "All-Hands Meeting This Friday",
                "Committee Meeting on Budget Planning", "Client Call Scheduled for 3 PM",
            ],
            "bodies": [
                "Meeting with the curriculum committee at 2 PM.",
                "Please join the Zoom session at 11 AM sharp.",
                "Postponed to 4 PM. New link attached.",
                "Kindly review the agenda before the meeting.",
                "The board meeting is scheduled in Conference Room B.",
                "Please confirm your attendance for the sync.",
                "Your one-on-one is set for Thursday at 3 PM.",
                "Google Meet link has been shared in the calendar invite.",
                "All department heads must attend this session.",
                "Minutes from last meeting have been shared via email.",
                "Standup call at 9 AM. Please be on time.",
                "Agenda and previous minutes are attached.",
            ],
        },
        "urgent": {
            "subjects": [
                "Immediate Action Required", "Submission Deadline Today",
                "URGENT: Please Respond", "Final Reminder: Approval Needed",
                "Critical Issue — Respond ASAP", "Last Date for Submission",
                "Action Needed by End of Day", "Emergency Notification",
                "High Priority: Pending Approval", "Overdue Task — Escalation",
                "Non-Compliance Warning", "Time-Sensitive Request",
            ],
            "bodies": [
                "Submit your report before 5 PM today.",
                "Final review required ASAP — Dean's sign-off needed.",
                "This is a time-critical issue. Please respond immediately.",
                "Your approval is pending. Non-compliance will be escalated.",
                "Deadline extended to tomorrow EOD. Submit now.",
                "This requires your immediate attention.",
                "Failure to respond will result in escalation.",
                "High priority task overdue by 2 days.",
                "Please complete the form before the deadline today.",
                "Emergency meeting called. Action required now.",
                "Last chance to respond before the deadline.",
                "Critical review required. Please act immediately.",
            ],
        },
        "personal": {
            "subjects": [
                "Lunch Plan This Friday?", "Congratulations on Your Promotion!",
                "Birthday Celebration Invite", "Family Reunion Next Month",
                "How are you doing?", "Weekend Trip Planning",
                "Checking In!", "Party Invite — Do Not Miss It",
                "Coffee Catch-Up?", "Greetings from the Team",
                "Vacation Photos!", "Hope You Are Doing Well",
            ],
            "bodies": [
                "Let us catch up over coffee after your session.",
                "Hope you are doing well! Miss you.",
                "Pictures from our event are attached.",
                "Are you free this Saturday for the reunion?",
                "Wishing you a very happy birthday!",
                "Let me know if you are joining the trip.",
                "Just checking in — hope everything is fine.",
                "We are celebrating this Friday at 7 PM.",
                "Sending warm wishes for the festival season.",
                "Congratulations on the new role!",
                "Vacation photos are attached. Hope you enjoy!",
                "It has been a while. Let us catch up soon.",
            ],
        },
        "information": {
            "subjects": [
                "New Guidelines for Exams", "Updated Faculty Handbook",
                "NAAC Accreditation Document", "Policy Update Notice",
                "Revised Evaluation Method", "Official Circular Released",
                "New Regulation Effective Immediately", "Release Notes v2.1",
                "Administrative Memo", "Newsletter — March Edition",
                "Revised Grading Policy", "Important Announcement",
            ],
            "bodies": [
                "Revised guideline document is attached for your reference.",
                "Updated protocol information has been sent to all faculty.",
                "Circulars regarding the new policy are on the portal.",
                "Please read the attached handbook before the next session.",
                "The new grading policy will be effective from next semester.",
                "Admin memo has been circulated. Please acknowledge receipt.",
                "New procedures are outlined in the attached PDF.",
                "This newsletter covers all updates for the quarter.",
                "The official circular has been released by the board.",
                "Exam policy changes are outlined in the attached document.",
                "This bulletin summarizes the key regulatory changes.",
                "Please review the release notes before the next sprint.",
            ],
        },
        "schedule": {
            "subjects": [
                "Next Week's Timetable", "Class Adjustment Notice",
                "Room Booking Confirmed", "Revised Exam Schedule",
                "Updated Invigilation Slots", "Lecture Timing Changed",
                "Duty Roster for April", "Office Hours Updated",
                "Session Rescheduled", "Availability for Next Week",
                "Exam Duty Assignment", "New Class Slot Allocated",
            ],
            "bodies": [
                "The proposed schedule for next week is attached.",
                "Please approve or suggest edits to the timetable.",
                "Invigilation slots have been uploaded on the portal.",
                "Your class on Monday has been moved to Wednesday.",
                "Room 204 is booked from 10 AM to 12 PM.",
                "Office hours updated: now Tuesdays 2 to 4 PM.",
                "The exam schedule has been revised — check the portal.",
                "New duty roster for April is attached.",
                "Your session has been rescheduled to 3 PM.",
                "Please confirm your availability for the time slot.",
                "Invigilation duty is assigned for 14th April.",
                "Your new class slot is Friday 11 AM to 1 PM.",
            ],
        },
        "security": {
            "subjects": [
                "Unusual Login Attempt Detected", "Account Security Alert",
                "Reset Your Password Immediately", "Suspicious Activity on Account",
                "Two-Factor Authentication Required", "New Device Login Detected",
                "Phishing Alert — Do Not Click", "Data Breach Notification",
                "Your Account May Be Compromised", "Verify Your Identity",
                "Unauthorized Access Blocked", "Security Policy Update",
            ],
            "bodies": [
                "Someone tried logging in from a new device.",
                "We noticed suspicious activity on your account.",
                "Change your password immediately to secure your account.",
                "Your account was accessed from an unrecognized location.",
                "Enable 2FA to protect your account from unauthorized access.",
                "A login attempt was blocked from an unknown IP address.",
                "Do not click on links in suspicious emails — phishing detected.",
                "Your data may have been exposed in a recent breach.",
                "Please verify your identity to restore account access.",
                "Security alert: OTP was requested from an unknown device.",
                "An unauthorized access attempt was blocked by our firewall.",
                "Please review the updated security policy attached.",
            ],
        },
    }
 
    data = []
    for category, tmpl in templates.items():
        for _ in range(n_per_class):
            subject = random.choice(tmpl["subjects"])
            body    = random.choice(tmpl["bodies"])
            data.append({"subject": subject, "body": body, "category": category})
 
    return pd.DataFrame(data)
 
 
# ─────────────────────────────────────────────
#  LOAD REAL EMAILS
# ─────────────────────────────────────────────
def load_real_emails():
    if os.path.exists("gmail_emails.csv"):
        df = pd.read_csv("gmail_emails.csv")
        df['subject'] = df['subject'].fillna('')
        df['body']    = df['body'].fillna('')
        if 'category' not in df.columns:
            df['category'] = df.apply(
                lambda row: auto_label(row['subject'], row['body']), axis=1
            )
        return df
    return pd.DataFrame(columns=["subject", "body", "category"])
 
 
# ─────────────────────────────────────────────
#  TRAIN & SAVE
# ─────────────────────────────────────────────
def build_and_train():
    print("📦 Generating synthetic training data...")
    df_syn  = generate_synthetic_emails(n_per_class=80)
    df_real = load_real_emails()
    df      = pd.concat([df_syn, df_real], ignore_index=True)
 
    df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    df = df.dropna(subset=['category'])
 
    print(f"✅ Total training samples: {len(df)}")
    print(df['category'].value_counts())
 
    # ── Vectorizer with bigrams ───────────────
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents='unicode',
        analyzer='word',
    )
 
    X = vectorizer.fit_transform(df['text'])
    y = df['category'].values
 
    # ── Optional SMOTE ────────────────────────
    if HAS_SMOTE:
        try:
            sm = SMOTE(random_state=42, k_neighbors=3)
            X, y = sm.fit_resample(X, y)
            print(f"✅ After SMOTE: {X.shape[0]} samples")
        except Exception as e:
            print(f"SMOTE skipped: {e}")
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    # ── Ensemble model ────────────────────────
    lr  = LogisticRegression(C=5.0, max_iter=1000, solver='lbfgs',
                             multi_class='multinomial', class_weight='balanced')
    svc = CalibratedClassifierCV(
              LinearSVC(C=1.0, max_iter=2000, class_weight='balanced'), cv=3)
    cnb = ComplementNB(alpha=0.3)
 
    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('svc', svc), ('cnb', cnb)],
        voting='soft',
        weights=[2, 2, 1],
    )
 
    print("\n🏋️ Training ensemble model (LR + SVM + Naive Bayes)...")
    ensemble.fit(X_train, y_train)
 
    y_pred = ensemble.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
 
    with open("email_classifier_model.pkl", "wb") as f:
        pickle.dump((vectorizer, ensemble), f)
    print("\n💾 Model saved → email_classifier_model.pkl")
 
    # ── Classify real emails if present ───────
    df_real_fresh = load_real_emails()
    if not df_real_fresh.empty:
        df_real_fresh['text'] = (df_real_fresh['subject'].fillna('') + ' '
                                 + df_real_fresh['body'].fillna(''))
        X_r = vectorizer.transform(df_real_fresh['text'])
        df_real_fresh['Predicted_Category'] = ensemble.predict(X_r)
 
        proba = ensemble.predict_proba(X_r)
        df_real_fresh['Confidence'] = np.max(proba, axis=1).round(3)
 
        df_real_fresh.to_csv("gmail_emails_classified.csv", index=False)
        print("📧 Classified emails saved → gmail_emails_classified.csv")
 
    return vectorizer, ensemble
 
 
# ─────────────────────────────────────────────
#  SINGLE EMAIL PREDICTION UTILITY
# ─────────────────────────────────────────────
def predict_email(subject, body, model_path="email_classifier_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run build_and_train() first.")
    with open(model_path, "rb") as f:
        vectorizer, model = pickle.load(f)
    text  = f"{subject} {body}"
    X     = vectorizer.transform([text])
    label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    conf  = dict(zip(model.classes_, proba.round(3)))
    return {"predicted": label, "confidence": conf}
 
 
# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    build_and_train()
 
    sample = predict_email(
        subject="Zoom meeting scheduled for tomorrow 10 AM",
        body="Please join. Agenda is attached in the calendar invite."
    )
    print(f"\n🔍 Sample prediction : {sample['predicted']}")
    print(f"   Confidence scores  : {sample['confidence']}")