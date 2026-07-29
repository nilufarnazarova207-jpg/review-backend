from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# Allow your frontend (running from a local file or any origin) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ... rest of your file stays the same

sentiment_vectorizer = joblib.load("sentiment_vectorizer.joblib")
sentiment_model = joblib.load("sentiment_model.joblib")
issue_vectorizer = joblib.load("issue_vectorizer.joblib")
issue_model = joblib.load("issue_model.joblib")
mlb = joblib.load("issue_label_binarizer.joblib")


class ReviewRequest(BaseModel):
    review: str


def analyze_review(review_text: str, issue_threshold: float = 0.3):
    if not review_text or len(review_text.strip()) < 3:
        return {"error": "Review text is empty or too short to analyze."}

    review_vec = sentiment_vectorizer.transform([review_text])
    sentiment_pred = sentiment_model.predict(review_vec)[0]
    sentiment_proba = sentiment_model.predict_proba(review_vec)[0]
    sentiment_confidence = max(sentiment_proba)
    sentiment_label = "positive" if sentiment_pred == 1 else "negative"

    review_vec2 = issue_vectorizer.transform([review_text])
    issue_probas = issue_model.predict_proba(review_vec2)[0]
    predicted_issues = [mlb.classes_[i] for i, p in enumerate(issue_probas) if p >= issue_threshold]
    if not predicted_issues:
        predicted_issues = ["general"]

    word_count = len(review_text.split())
    low_confidence_flag = bool(word_count < 5 or sentiment_confidence < 0.6)

    return {
        "review": review_text,
        "sentiment": sentiment_label,
        "sentiment_confidence": round(float(sentiment_confidence), 2),
        "issue_categories": predicted_issues,
        "low_confidence_warning": low_confidence_flag,
    }


@app.post("/analyze")
def analyze(request: ReviewRequest):
    return analyze_review(request.review)


@app.get("/")
def root():
    return {"status": "Review Intelligence API is running"}