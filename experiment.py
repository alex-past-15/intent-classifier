"""Подготовка данных, обучение моделей и проверка качества на BANKING77."""
import argparse
import hashlib
import json
import random
import re
import time
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
DATA_REV = "57ec275d8078af65b7731c2a98be812d844a6d6b"
MODEL_REV = "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"
MODEL_ID = "prajjwal1/bert-tiny"
SEED = 42
TARGET_ACCURACY = .95


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with requests.get(url, timeout=(10, 60), stream=True) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
    temporary.replace(path)


def normalize(text):
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def prepare_data():
    raw = {}
    provenance = {}
    for split in ["train", "test"]:
        url = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{DATA_REV}/banking_data/{split}.csv"
        path = DATA / "raw" / f"{split}.csv"
        download(url, path)
        frame = pd.read_csv(path).rename(columns={"category": "label"})
        if not {"text", "label"}.issubset(frame.columns) or frame[["text", "label"]].isna().any().any():
            raise ValueError("Unexpected dataset schema or missing values")
        frame["key"] = frame.text.map(normalize)
        raw[split] = frame
        provenance[split] = {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(frame)}
    # Убираем из обучающих данных сообщения, которые есть в тесте.
    train = raw["train"]
    conflicts = train.groupby("key").label.nunique()
    train = train[~train.key.isin(conflicts[conflicts > 1].index)]
    train = train.drop_duplicates("key")
    train = train[~train.key.isin(raw["test"].key)]
    fit, holdout = train_test_split(train, test_size=.30, stratify=train.label, random_state=SEED)
    validation, calibration = train_test_split(holdout, test_size=.50, stratify=holdout.label, random_state=SEED)
    frames = {"train": fit, "validation": validation, "calibration": calibration, "test": raw["test"]}
    for name, frame in frames.items():
        frame.to_csv(DATA / f"{name}.csv", index=False)
    for a, b in [("train", "validation"), ("train", "calibration"), ("validation", "calibration"),
                 ("train", "test"), ("validation", "test"), ("calibration", "test")]:
        assert set(frames[a].key).isdisjoint(frames[b].key)
    combined = pd.concat([f.assign(split=k) for k, f in frames.items()], ignore_index=True)
    counts = pd.crosstab(combined.label, combined.split)
    REPORTS.mkdir(exist_ok=True)
    counts.to_csv(REPORTS / "class_counts.csv")
    save_json(REPORTS / "data_audit.json", {
        "dataset": "BANKING77", "revision": DATA_REV, "license": "CC-BY-4.0", "language": "English",
        "source": provenance, "splits": {k: len(v) for k, v in frames.items()},
        "classes": int(train.label.nunique()), "removed_from_train": len(raw["train"]) - len(train),
        "conflicting_train_keys": int((conflicts > 1).sum()),
        "test_duplicate_keys": int(raw["test"].key.duplicated().sum()),
        "normalization": "NFKC, casefold, punctuation/whitespace normalization",
        "limitations": "Near-duplicate paraphrases are not removed. Official test rows are preserved.", "seed": SEED,
    })
    return frames


def choose_threshold(y, probabilities, classes, target=TARGET_ACCURACY, min_accepted=50):
    confidence = probabilities.max(axis=1)
    correct = np.asarray(classes)[probabilities.argmax(axis=1)] == np.asarray(y)
    for threshold in np.unique(confidence):
        accepted = confidence >= threshold
        if accepted.sum() >= min_accepted and correct[accepted].mean() >= target:
            return float(threshold)
    return None  # Подходящего порога нет, все сообщения проверяет человек.


def selective_metrics(y, probabilities, classes, threshold):
    accepted = np.zeros(len(y), dtype=bool) if threshold is None else probabilities.max(axis=1) >= threshold
    correct = np.asarray(classes)[probabilities.argmax(axis=1)] == np.asarray(y)
    n = int(accepted.sum())
    return {"coverage": float(accepted.mean()), "accepted": n,
            "accepted_accuracy": float(correct[accepted].mean()) if n else None,
            "manual_review": int(len(y) - n)}


def evaluate(name, frames, predictions, classes, metadata):
    cal = frames["calibration"]
    threshold = choose_threshold(cal.label, predictions["calibration"], classes)
    test = frames["test"]
    proba = predictions["test"]
    predicted = np.asarray(classes)[proba.argmax(axis=1)]
    report = {"model": name, **metadata, "threshold": threshold, "target_accuracy": TARGET_ACCURACY,
              "test_accuracy": float(accuracy_score(test.label, predicted)),
              "test_macro_f1": float(f1_score(test.label, predicted, average="macro")),
              "calibration": selective_metrics(cal.label, predictions["calibration"], classes, threshold),
              "test": selective_metrics(test.label, proba, classes, threshold)}
    save_json(REPORTS / f"{name}.json", report)
    table = test[["text", "label"]].copy()
    table["predicted"] = predicted
    table["confidence"] = proba.max(axis=1)
    table["accepted"] = False if threshold is None else table.confidence >= threshold
    table.to_csv(REPORTS / f"{name}_predictions.csv", index=False)
    table[table.label != table.predicted].sort_values("confidence", ascending=False).to_csv(REPORTS / f"{name}_errors.csv", index=False)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def train_baseline(frames):
    started = time.monotonic()
    best_score, best, experiments = -1, None, []
    for c in [1., 4., 16.]:
        model = Pipeline([
            ("features", FeatureUnion([
                ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=35000)),
                ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=35000)),
            ])),
            ("classifier", LogisticRegression(C=c, max_iter=600, random_state=SEED)),
        ])
        with threadpool_limits(limits=4):
            model.fit(frames["train"].text, frames["train"].label)
        score = f1_score(frames["validation"].label, model.predict(frames["validation"].text), average="macro")
        experiments.append({"C": c, "validation_macro_f1": float(score)})
        print("TF-IDF", experiments[-1], flush=True)
        if score > best_score:
            best_score, best = score, model
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(best, ARTIFACTS / "tfidf.joblib")
    predictions = {s: best.predict_proba(frames[s].text) for s in ["calibration", "test"]}
    return evaluate("tfidf", frames, predictions, best.classes_, {"seconds": time.monotonic() - started, "experiments": experiments})


def tiny_path():
    folder = ARTIFACTS / "bert-tiny-pretrained"
    for file in ["config.json", "vocab.txt", "pytorch_model.bin"]:
        download(f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REV}/{file}?download=true", folder / file)
    return folder


def train_transformer(frames, epochs=35):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    started = time.monotonic()
    pretrained = tiny_path()
    labels = sorted(frames["train"].label.unique())
    label_to_id = {label: i for i, label in enumerate(labels)}
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    config = BertConfig.from_pretrained(pretrained, num_labels=len(labels),
                    id2label=dict(enumerate(labels)), label2id=label_to_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batches = {}
    for name, frame in frames.items():
        encoded = tokenizer(frame.text.tolist(), padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], torch.tensor([label_to_id[x] for x in frame.label]))
        batches[name] = DataLoader(dataset, batch_size=32 if name == "train" else 64, shuffle=name == "train", num_workers=0)

    def predict(split):
        model.eval()
        values = []
        with torch.inference_mode():
            for ids, mask, _ in batches[split]:
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                values.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(values)

    previous_path = REPORTS / "bert_tiny.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else None
    best_score, best_state, selected = -1, None, None
    trials = []
    # Выбираем настройки по validation, тест используем после выбора модели.
    for learning_rate in [1e-4, 3e-4]:
        torch.manual_seed(SEED)
        model = BertForSequenceClassification.from_pretrained(pretrained, config=config).to(device)
        optimizer = torch.optim.AdamW([
            {"params": model.bert.parameters(), "lr": learning_rate},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ], weight_decay=.01)
        total_steps = epochs * len(batches["train"])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(.1 * total_steps),
                                                   num_training_steps=total_steps)
        trial_best, stale_epochs, history = -1, 0, []
        for epoch in range(epochs):
            model.train()
            losses = []
            for ids, mask, target in batches["train"]:
                optimizer.zero_grad(set_to_none=True)
                loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=target.to(device)).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
            val = predict("validation")
            score = f1_score(frames["validation"].label, np.asarray(labels)[val.argmax(axis=1)], average="macro")
            history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_macro_f1": float(score)})
            print("BERT", learning_rate, history[-1], flush=True)
            if score > best_score:
                best_score = score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                selected = {"learning_rate": learning_rate, "best_epoch": epoch + 1}
            if score > trial_best + .001:
                trial_best, stale_epochs = score, 0
            else:
                stale_epochs += 1
            if stale_epochs >= 5:
                break
        trials.append({"learning_rate": learning_rate, "history": history})
    model.load_state_dict(best_state)
    model.save_pretrained(ARTIFACTS / "bert-tiny")
    tokenizer.save_pretrained(ARTIFACTS / "bert-tiny")
    predictions = {s: predict(s) for s in ["calibration", "test"]}
    history = next(t["history"] for t in trials if t["learning_rate"] == selected["learning_rate"])
    return evaluate("bert_tiny", frames, predictions, labels, {
        "seconds": time.monotonic() - started, "device": device, "model_id": MODEL_ID,
        "revision": MODEL_REV, **selected, "history": history, "trials": trials,
        "validation_macro_f1": float(best_score), "max_epochs": epochs, "early_stopping_patience": 5,
        "early_stopping_min_delta": .001, "classifier_learning_rate": 1e-3,
        "schedule": "10% warmup then linear decay", "max_tokens": 64, "batch_size": 32,
        "training": "full fine-tuning", "test_status": "Reused benchmark test; tuning uses validation only, not a new blind holdout.",
        "previous_run": ({k: previous[k] for k in ["test_macro_f1", "test_accuracy", "test", "best_epoch"]} if previous else None),
    })



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tfidf", "bert", "all", "prepare"], default="all")
    parser.add_argument("--epochs", type=int, default=35)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    data = prepare_data()
    if args.model in ["tfidf", "all"]:
        train_baseline(data)
    if args.model in ["bert", "all"]:
        train_transformer(data, args.epochs)
