from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        news = request.form["news"]

        # SAME as predictor.py
        news = news.strip()

        vector = vectorizer.transform([news])
        result = model.predict(vector)

        if result[0] == 1:
            prediction = "🟢 Real News"
        else:
            prediction = "🔴 Fake News"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)