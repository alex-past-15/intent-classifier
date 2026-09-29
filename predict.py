"""Определение темы сообщения с помощью обученной модели."""
import argparse
import json

import joblib
import numpy as np

from experiment import ARTIFACTS, REPORTS


class Predictor:
    def __init__(self, name="tfidf"):
        if name not in {"tfidf", "bert_tiny"}:
            raise ValueError("Unknown model")
        self.name = name
        self.report = json.loads((REPORTS / f"{name}.json").read_text(encoding="utf-8"))
        if name == "tfidf":
            self.model = joblib.load(ARTIFACTS / "tfidf.joblib")
            self.labels = self.model.classes_
        else:
            import torch
            from transformers import AutoModelForSequenceClassification, BertTokenizer
            torch.set_num_threads(4)
            self.tokenizer = BertTokenizer.from_pretrained(ARTIFACTS / "bert-tiny")
            self.model = AutoModelForSequenceClassification.from_pretrained(ARTIFACTS / "bert-tiny").eval()
            self.labels = np.array([self.model.config.id2label[i] for i in range(self.model.config.num_labels)])

    def predict(self, text):
        if not text.strip() or len(text) > 3000:
            raise ValueError("Введите непустое обращение до 3000 символов.")
        if self.name == "tfidf":
            proba = self.model.predict_proba([text])[0]
        else:
            import torch
            inputs = self.tokenizer(text, truncation=True, max_length=self.report["max_tokens"], return_tensors="pt")
            with torch.inference_mode():
                proba = torch.softmax(self.model(**inputs).logits, dim=-1)[0].numpy()
        indexes = np.argsort(proba)[::-1][:3]
        threshold = self.report["threshold"]
        accepted = threshold is not None and float(proba[indexes[0]]) >= threshold
        return {"decision": "automatic" if accepted else "manual_review", "threshold": threshold,
                "top3": [{"label": str(self.labels[i]), "score": float(proba[i])} for i in indexes]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--model", choices=["tfidf", "bert_tiny"], default="tfidf")
    args = parser.parse_args()
    print(json.dumps(Predictor(args.model).predict(args.text), ensure_ascii=False, indent=2))
