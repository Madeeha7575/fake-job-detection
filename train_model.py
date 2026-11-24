import pandas as pd

df = pd.read_csv("fake_real_job_postings_3000x25.csv")
import re

from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))


def clean_text(text):
    text=str(text)
    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()
    words=[word for word in words if word not in stop_words]
    return " ".join(words)
df["cleaned_description"] = df["job_description"].apply(clean_text)
print(df["cleaned_description"].iloc[0])

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(
    df["cleaned_description"]
)

y = df["is_fake"]

print(x.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x
,y,test_size=0.2,random_state=42)
print(x_train.shape)
print(y_train.shape)

from sklearn.linear_model import LogisticRegression
model =LogisticRegression()
model.fit(x_train,y_train)
predictions= model.predict(x_test)
print(predictions[:20])

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predictions)
print("Accuracy:",accuracy) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
print(cm)
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        predictions
    )
)
import matplotlib.pyplot as plt 
import seaborn as sns

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
"confusion_matrix.png",
dpi=300,
bbox_inches="tight"
)

plt.show()
import joblib

joblib.dump(
    model,
    "fake_job_model.pkl"
)

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)

print("Model saved successfully!")