import requests
from bs4 import BeautifulSoup
import json

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
            return None
        soup = BeautifulSoup(res.text, 'html.parser')
    except Exception:
        return None

    title = get_text_or_none(soup.select_one('h1.article__headline')) or get_text_or_none(soup.select_one('h1'))
    abstract = get_text_or_none(soup.select_one('div.title > p'))

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

    article = soup.select_one('article.full-article')
    text_fragments = []

    if article:
        for child in article.children:
            if not hasattr(child, 'name'):
                continue
            if child.name in ('h1', 'h2', 'p', 'blockquote'):
                text = get_text_or_none(child)
                if text:
                    text_fragments.append(text)
            elif child.name == 'ul':
                for li in child.find_all('li', recursive=False):
                    text = get_text_or_none(li)
                    if text:
                        text_fragments.append(text)

    full_text = "\n\n".join(text_fragments) if text_fragments else None

    return {
        "title": title,
        "abstract": abstract,
        "publish_date": publish_date,
        "publish_time": publish_time,
        "update_date": update_date,
        "update_time": update_time,
        "text": full_text,
    }


with open('../Data/cryptoslate_links_2y.json', 'r', encoding='utf-8') as f:
    url_items = json.load(f)

all_data = []
batch_size = 50
file_index = 1

for i, item in enumerate(url_items, 1):
    url = item.get("link")
    label = item.get("label")
    print(f"Scraping {url} ...")
    article_data = scrape_article(url)
    
    if article_data is None:
        article_data = {
            "title": None,
            "abstract": None,
            "publish_date": None,
            "publish_time": None,
            "update_date": None,
            "update_time": None,
            "text": None,
        }
    
    article_data.update({
        "url": url,
        "label": label,
    })
    all_data.append(article_data)

    # Every 50 URLs, save current batch to file and clear list
    if i % batch_size == 0:
        output_filename = f'articles_data_with_labels_part{file_index}.json'
        with open(output_filename, 'w', encoding='utf-8') as f_out:
            json.dump(all_data, f_out, ensure_ascii=False, indent=2)
        print(f"Saved {len(all_data)} records to {output_filename}")
        all_data = []
        file_index += 1

# Save any remaining data after loop ends
if all_data:
    output_filename = f'articles_data_with_labels_part{file_index}.json'
    with open(output_filename, 'w', encoding='utf-8') as f_out:
        json.dump(all_data, f_out, ensure_ascii=False, indent=2)
    print(f"Saved remaining {len(all_data)} records to {output_filename}")

print("Done! Data saved in batches.")
