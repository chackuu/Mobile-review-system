
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("Amazon_Unlocked_Mobile.csv")

# Remove missing reviews and ratings
df = df.dropna(subset=["Reviews", "Rating"])

# Remove neutral 3-star reviews
df = df[df["Rating"] != 3].copy()

# Create sentiment labels
df["Sentiment"] = df["Rating"].apply(
    lambda rating: "Positive" if rating >= 4 else "Negative"
)

# Convert review text to strings
df["Reviews"] = df["Reviews"].astype(str)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    df["Reviews"],
    df["Sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["Sentiment"]
)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# Convert text into TF-IDF features
tfidf = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and TF-IDF vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\nModel and TF-IDF vectorizer saved successfully!")

