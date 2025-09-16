import json
import ollama
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

# === Config ===
MAX_WORKERS = 2           # Adjust based on system/load

lock = Lock()  # For thread-safe file writing


def extract_json_from_response(response_text):
    match = re.search(r'{[\s\S]*?}', response_text)
    if match:
        json_str = match.group(0)
        json_str = re.sub(r':\s*\+([0-9.]+)', r': \1', json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
    else:
        print("No JSON object found in response.")
    return None


def estimate_btc_impact(title, abstract, text, label):
    prompt = f"""
You are a financial analyst. Read the news article below and assess how it might affect the price of Bitcoin (BTC).
Give a numeric impact score between -1.0 (strong negative impact) to +1.0 (strong positive impact), and provide a one-sentence summary of why.

Title: {title if title else "No title provided"}
Abstract: {abstract if abstract else "No abstract provided"}
Label: {label if label else "No label provided"}
Text: {text if text else "No text provided"}

Respond in this JSON format:
{{
  "impact_score": float,
  "impact_summary": string
}}
"""
    try:
        response = ollama.chat(
            model='llama3.1:latest',
            messages=[{"role": "user", "content": prompt}]
        )
        content = response['message']['content']
        parsed = extract_json_from_response(content)
        if parsed:
            return parsed.get("impact_score", 0.0), parsed.get("impact_summary", "")
        else:
            return 0.0, "Could not analyze response format."
    except Exception as e:
        print("Error calling ollama or parsing response:", e)
        return 0.0, "Could not analyze due to exception."


def analyze_article(article, processed_urls, RESULT_FILE):
    url = article.get("url", "")
    if url in processed_urls:
        return None  # Skip already processed

    title = article.get("title", "")
    abstract = article.get("abstract", "")
    text = article.get("text", "")
    label = article.get("label", "")
    publish_date = article.get("publish_date", "")
    publish_time = article.get("publish_time", "")
    update_date = article.get("update_date", "")
    update_time = article.get("update_time", "")

    print(f"Analyzing: {title[:60]}...")

    score, summary = estimate_btc_impact(title, abstract, text, label)

    result = {
        "url": url,
        "label": label,
        "publish_date": publish_date,
        "publish_time": publish_time,
        "update_date": update_date,
        "update_time": update_time,
        "impact_score": score,
        "impact_summary": summary,
    }

    with lock:
        with open(RESULT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result


def load_processed_urls(result_file):
    urls = set()
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    urls.add(record.get("url"))
                except json.JSONDecodeError:
                    continue
    return urls


def main(INPUT_FILE, RESULT_FILE):
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)

    processed_urls = load_processed_urls(RESULT_FILE)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(analyze_article, article, processed_urls, RESULT_FILE)
            for article in articles
        ]

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing articles"):
            pass

    print(f"\nFinished analysis. Results written to {RESULT_FILE}.")


if __name__ == "__main__":
    for i in range(1, 3):
        INPUT_FILE = f"articles_data_with_labels_part{i}.json"
        RESULT_FILE = f"articles_data_part{i}_result.json"
        main(INPUT_FILE, RESULT_FILE)