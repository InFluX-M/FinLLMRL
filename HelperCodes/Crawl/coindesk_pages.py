import requests
from lxml import html
from datetime import datetime, timedelta
import re
import json 

def normalize_datetime_string(raw: str) -> str:
    """
    Convert Coindesk-style datetime string into 'YYYY-MM-DD HH:MM' 24-hour format.
    Handles formats like:
      - 'Published May 22, 2025, 8:06 a.m.'
      - 'Updated May 22, 2025, 12:58 p.m.'
      - 'May 22, 2025, 1:36 p.m.'
    """
    if not raw:
        return ""

    # Normalize spaces and strip leading label
    cleaned = raw.replace('\u202f', ' ').replace('\xa0', ' ').strip()
    cleaned = re.sub(r'^(Published|Updated)\s+', '', cleaned, flags=re.IGNORECASE)

    # Normalize a.m. / p.m. to AM / PM
    cleaned = cleaned.lower().replace('a.m.', 'AM').replace('p.m.', 'PM')

    try:
        dt = datetime.strptime(cleaned, "%B %d, %Y, %I:%M %p")
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ""

def add_3h30_to_str(datetime_str: str) -> str:
    """
    Given a datetime string in '%Y-%m-%d %H:%M' format,
    add 3 hours and 30 minutes, and return as string.
    Return empty string if input invalid.
    """
    if not datetime_str:
        return ""

    try:
        dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
        dt_plus = dt + timedelta(hours=3, minutes=30)
        return dt_plus.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ""

def extract_coindesk_article_with_dates(url: str) -> dict:
    """
    Extracts article text, published date, and updated date from a Coindesk article,
    and also returns those dates plus 3h30 minutes.
    Returns a dict with keys:
      - 'text'
      - 'published'
      - 'updated'
    """
    response = requests.get(url)
    response.raise_for_status()

    tree = html.fromstring(response.content)

    # Extract article content
    content_root = tree.xpath('/html/body/div[1]/main/section/div')
    if not content_root:
        return {"text": "", "published": "", "published_plus_3h30": "",
                "updated": "", "updated_plus_3h30": ""}

    content_div = content_root[0]
    nodes = content_div.xpath('.//h1 | .//h2 | .//h3 | .//li | .//p')
    lines = [node.text_content().strip() for node in nodes if node.text_content().strip()]
    article_text = "\n".join(lines)

    # Extract updated and published date/time
    span_nodes = tree.xpath('/html/body/div[1]/main/section/div/div[2]/div/div[2]/span')
    raw_updated = span_nodes[0].text_content().strip() if len(span_nodes) >= 1 else ""
    raw_published = span_nodes[1].text_content().strip() if len(span_nodes) >= 2 else ""

    updated_norm = normalize_datetime_string(raw_updated)
    published_norm = normalize_datetime_string(raw_published)

    return {
        "text": article_text,
        "published": add_3h30_to_str(published_norm),
        "updated": add_3h30_to_str(updated_norm),
    }

import json

# Assuming your earlier functions are already defined:
# - normalize_datetime_string
# - add_3h30_to_str
# - extract_coindesk_article_with_dates

def enrich_articles(json_file_path: str, output_path: str):
    # Load the JSON list
    with open(json_file_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    enriched_articles = []

    for article in articles:
        url = article.get('url', '')
        title = article.get('title', '')
        if not url:
            continue

        try:
            extracted = extract_coindesk_article_with_dates(url)
            enriched_articles.append({
                "title": title,
                "url": url,
                "text": extracted.get("text", ""),
                "published": extracted.get("published", ""),
                "updated": extracted.get("updated", "")
            })
        except Exception as e:
            print(f"Failed to process {url}: {e}")
            enriched_articles.append({
                "title": title,
                "url": url,
                "text": "",
                "published": "",
                "updated": ""
            })

    # Save the enriched articles to output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_articles, f, ensure_ascii=False, indent=2)

# Example usage
enrich_articles("links.json", "news.json")