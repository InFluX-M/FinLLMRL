from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import json

options = Options()
options.add_argument('--headless')  # Optional
driver = webdriver.Chrome(options=options)

articles_data = []

try:
    for page in range(1, 475):
        url = f'https://cryptoslate.com/news/page/{page}'
        print(f'Visiting: {url}')
        driver.get(url)
        time.sleep(2)

        articles = driver.find_elements(By.XPATH, '//section[1]//article/a')

        for a in articles:
            link = a.get_attribute('href')

            # Find nested span/span[1]
            try:
                label_elem = a.find_element(By.XPATH, './/div[2]/div/span/span[1]')
                label = label_elem.text.strip()
            except:
                label = None

            articles_data.append({
                "link": link,
                "label": label.lower() if label else None
            })

except Exception as e:
    print(f"Error: {e}")

finally:
    driver.quit()

# Deduplicate by link
unique_articles = {item['link']: item for item in articles_data}.values()

# Save to JSON
with open("cryptoslate_articles.json", "w") as f:
    json.dump(list(unique_articles), f, indent=2)

print(f"\n✅ Saved {len(unique_articles)} articles with labels to cryptoslate_articles.json")
