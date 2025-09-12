from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import json 

# Load pretrained FinBERT
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def finbert_sentiment(subject, body):
    text = subject + " " + body if body else subject
    text = text.strip()

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Run model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get probabilities
    probs = F.softmax(outputs.logits, dim=-1)

    # Labels: [0: positive, 1: negative, 2: neutral]
    labels = ["positive", "negative", "neutral"]
    pred = torch.argmax(probs, dim=1).item()

    return {
        "finbert_label": labels[pred],
        "finbert_sentiment": {labels[i]: probs[0][i].item() for i in range(len(labels))}
    }

def add_finbert(stock):
    # Load data
    with open(f"new_{stock}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    idx = 0
    for item in data:
        headline = item.get("headline", "")
        summary = item.get("summary", "")

        if "finbert_label" not in item:
            # Run FinBERT
            result = finbert_sentiment(headline, summary)
            item["finbert_label"] = result["finbert_label"]
            item["finbert_score"] = result["finbert_sentiment"]

        # Delete old fields if they exist
        item.pop("sentiment", None)
        item.pop("sentiment_score", None)

        idx += 1

        if idx % 100 == 0:
            print(f"Processed {idx} articles")

    # Save back to file
    with open(f"new_{stock}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Example usage
if __name__ == "__main__":
    path = "GS"
    add_finbert(path)