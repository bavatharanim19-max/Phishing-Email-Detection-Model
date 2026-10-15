
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Dataset: 1 = Phishing, 0 = Safe
data = {
    "email": [
        "Urgent verify your bank account immediately",
        "Congratulations you won a free prize click now",
        "Your account is suspended reset your password",
        "Claim your lottery reward by clicking this link",
        "Update payment details immediately to avoid blocking",
        "You have won a gift card claim your reward",
        "Click here to verify your account password",
        "Security alert login now to prevent suspension",
        "Urgent confirm your bank details",
        "Free cash reward available claim now",
        "Your account has been compromised verify now",
        "Click this link to receive your prize",
        "Meeting scheduled for tomorrow morning",
        "Please find the attached college assignment",
        "Your order has been shipped successfully",
        "Hello please call me when you are free",
        "The class timetable is available online",
        "Thank you for attending the meeting",
        "Your library book is due next week",
        "The project submission deadline is Friday",
        "Your appointment is confirmed for Monday",
        "Please review the report before the meeting",
        "The exam results will be announced tomorrow",
        "Your course registration is complete",
        "http://192.168.1.5/login verify your bank password",
        "http://free-prize.example claim your cash reward",
        "Click http://account-check.example to restore access",
        "http://secure-login.example confirm your credentials",
        "https://college.example your class starts at 10 AM",
        "https://shop.example your order is confirmed",
        "https://library.example your book is due soon",
        "https://school.example download your class notes"
    ],
    "label": [
        "Phishing", "Phishing", "Phishing", "Phishing",
        "Phishing", "Phishing", "Phishing", "Phishing",
        "Phishing", "Phishing", "Phishing", "Phishing",
        "Safe", "Safe", "Safe", "Safe",
        "Safe", "Safe", "Safe", "Safe",
        "Safe", "Safe", "Safe", "Safe",
        "Phishing", "Phishing", "Phishing", "Phishing",
        "Safe", "Safe", "Safe", "Safe"
    ]
}

df = pd.DataFrame(data)

# Extract URL and suspicious keyword features
keywords = [
    "urgent", "verify", "password", "prize",
    "claim", "suspended", "reward", "credentials",
    "bank", "cash", "compromised"
]

def extract_features(email):
    email = email.lower()

    urls = re.findall(r'https?://\S+', email)
    keyword_count = sum(email.count(word) for word in keywords)
    has_ip_url = int(bool(re.search(
        r'https?://(?:\d{1,3}\.){3}\d{1,3}', email
    )))
    has_short_url = int(bool(re.search(
        r'https?://(?:bit\.ly|tinyurl\.com|t\.co)/', email
    )))

    # Add extracted features to email text
    return (
        email
        + " URL_COUNT_" + str(min(len(urls), 3))
        + " KEYWORD_COUNT_" + str(min(keyword_count, 5))
        + " HAS_IP_URL_" + str(has_ip_url)
        + " HAS_SHORT_URL_" + str(has_short_url)
    )

df["features"] = df["email"].apply(extract_features)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["features"],
    df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)

# Build and train the machine learning model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

# Predict test emails
y_pred = model.predict(X_test)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# Display confusion matrix
labels = ["Phishing", "Safe"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

print("Confusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)
display.plot(cmap="Blues", values_format="d")
plt.title("Phishing Email Detection - Confusion Matrix")
plt.show()

# Test a new email
new_email = input("\nEnter an email to check: ")
processed_email = extract_features(new_email)
prediction = model.predict([processed_email])[0]

print("\nEmail Classification:", prediction)

if prediction == "Phishing":
    print("Warning: This email may be a phishing attempt!")
else:
    print("This email appears safe, but verify links before clicking.")
