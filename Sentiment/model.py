import json
import ollama
import re
import os

def extract_json_from_response(response_text):
    """
    Extract and clean JSON-like content from the LLM response.
    Removes backticks and fixes non-standard JSON (e.g., +0.3).
    """
    match = re.search(r'{[\s\S]*?}', response_text)
    if match:
        json_str = match.group(0)

        # Remove + signs in numbers (e.g., +0.3)
        json_str = re.sub(r':\s*\+([0-9.]+)', r': \1', json_str)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
    else:
        print("No JSON object found in response.")
    return None


def estimate_btc_impact(text):
    prompt = f"""
You are a financial analyst. Read the news article below and assess how it might affect the price of Bitcoin (BTC). 
Give a numeric impact score between -1.0 (strong negative impact) to +1.0 (strong positive impact), and provide a one-sentence summary of why.

News article:
{text[:4000]}

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


# Load news articles
with open("news.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

# Ensure result file exists (optional: clear or keep existing content)
result_file = "result.json"
if not os.path.exists(result_file):
    open(result_file, "w", encoding="utf-8").close()

# Analyze and append results
with open(result_file, "a", encoding="utf-8") as f:
    for article in articles:
        title = article.get("title", "")
        text = article.get("text", "")
        url = article.get("url", "")
        date = article.get("date", "")

        print(f"Analyzing: {title[:60]}...")
        score, summary = estimate_btc_impact(text)

        result = {
            "title": title,
            "date": date,
            "url": url,
            "impact_score": score,
            "impact_summary": summary,
            "text": text  # Remove if you want a smaller result file
        }

        # Write each result as one JSON line
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print("Finished analysis. Results appended to result.json.")
