
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained machine learning model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    review = ""

    if request.method == "POST":
        review = request.form.get("review", "").strip()

        if review:
            review_tfidf = tfidf.transform([review])
            prediction = model.predict(review_tfidf)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        review=review
    )


if __name__ == "__main__":
    app.run(debug=True)
