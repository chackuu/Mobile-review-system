
import joblib

# Load the saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Enter a sample mobile review
review = ["This phone has terrible battery life and a poor camera"]

# Convert the review into numerical features
review_tfidf = tfidf.transform(review)

# Predict the sentiment
prediction = model.predict(review_tfidf)

print("Review:", review[0])
print("Predicted Sentiment:", prediction[0])
