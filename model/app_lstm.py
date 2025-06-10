# app.py
from fastapi import FastAPI
from lstm_inference import default_classifier
from pydantic import BaseModel
from typing import List

app = FastAPI()
model = default_classifier()

class NewsRequest(BaseModel):
    articles: List[str]

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return 'Success', 200

@app.post("/predict")
def predict(request: NewsRequest):
    predictions = model.predict_label(request.articles)
    return {
        "predictions": [
            {"article": text, "class": label}
            for text, (label, _) in zip(request.articles, predictions)
        ]
    }

# request = NewsRequest(articles=["The stock market is booming today.", "New advancements in AI technology."])
# predictions = predict(request)
# print(predictions)