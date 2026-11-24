import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv("fake_real_job_postings_3000x25.csv")

# Clean text and remove stopwords
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text)
    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["cleaned_description"] = (
    df["job_description"]
    .apply(clean_text)
)

# Convert text into numerical features
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(
    df["cleaned_description"]
)

y = df["is_fake"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression()

model.fit(
    X_train,
    y_train
)

# Make predictions
predictions = model.predict(
    X_test
)

# Evaluate model
accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)

print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        predictions
    )
)

# Visualize confusion matrix
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Fake Job Detection Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "confusion
