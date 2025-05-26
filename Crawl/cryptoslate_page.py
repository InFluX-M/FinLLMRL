import requests
from bs4 import BeautifulSoup
import json

def get_text_or_none(el):
    return el.text.strip() if el else None

import requests
from bs4 import BeautifulSoup

def get_text_or_none(element):
    try:
        if element:
            text = element.get_text(strip=True)
            return text if text else None
    except Exception:
        pass
    return None

def scrape_article(url):
    try:
        res = requests.get(url)
        if res.status_code != 200:
            return None  # Or raise an error or handle differently
        soup = BeautifulSoup(res.text, 'html.parser')
    except Exception:
        return None

    # Title and abstract
    title = get_text_or_none(soup.select_one('h1.article__headline')) or get_text_or_none(soup.select_one('h1'))
    abstract = get_text_or_none(soup.select_one('div.title > p'))

    # Publish date and time
    publish_div = soup.select_one('div.post-date')
    publish_date = None
    publish_time = None
    if publish_div:
        try:
            text = publish_div.contents[0].strip()
            publish_date = text if text else None
        except (IndexError, AttributeError):
            publish_date = None
        publish_time = get_text_or_none(publish_div.find('span', class_='time'))

    # Update date and time
    update_div = soup.select_one('div.post-reading div.post-date')
    update_date = None
    update_time = None
    if update_div:
        try:
            updated_label = update_div.find('span', class_='break')
            if updated_label and updated_label.next_sibling:
                text = updated_label.next_sibling.strip()
                update_date = text if text else None
        except Exception:
            update_date = None
        update_time = get_text_or_none(update_div.find('span', class_='time'))

    return {
        "title": title,
        "abstract": abstract,
        "publish_date": publish_date,
        "publish_time": publish_time,
        "update_date": update_date,
        "update_time": update_time,
    }

# Load URLs and labels from JSON file (replace 'input_urls.json' with your filename)
with open('cryptoslate_articles.json', 'r', encoding='utf-8') as f:
    url_items = json.load(f)

all_data = []

for item in url_items:
    url = item.get("link")
    label = item.get("label")
    print(f"Scraping {url} ...")
    article_data = scrape_article(url)
    
    # Combine label and URL info with scraped data
    article_data.update({
        "url": url,
        "label": label,
    })
    all_data.append(article_data)

# Save result to output JSON
with open('articles_data_with_labels.json', 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print("Done! Data saved to articles_data_with_labels.json")
