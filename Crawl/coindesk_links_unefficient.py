from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import json

# Set up Selenium options
options = Options()
# options.add_argument("--headless")  # Run in headless mode
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Initialize WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Open the CoinDesk Latest Crypto News page
url = "https://www.coindesk.com/markets"
driver.get(url)

# Initialize variables
articles_seen = set()
scroll_pause_time = 2

cnt = 0

while True:
    # Get current number of articles
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    articles = soup.find_all('a', href=True, title=True)
    current_count = len(articles)

    try:
        # Scroll to the bottom to load new content
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)

        # Wait for the "More stories" button to be clickable
        button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'More stories')]"))
        )

        # Click the button using ActionChains
        ActionChains(driver).move_to_element(button).click().perform()
        print("Clicked 'More stories' button.")
        cnt += 1
        time.sleep(scroll_pause_time)

        if cnt > 1000:
            break

        # Wait until new articles are loaded
        WebDriverWait(driver, 30).until(
            lambda d: len(BeautifulSoup(d.page_source, 'html.parser').find_all('a', href=True, title=True)) > current_count
        )

    except Exception as e:
        print("No more 'More stories' button found or no new articles loaded.")
        break

# Parse the final page content
soup = BeautifulSoup(driver.page_source, 'html.parser')
driver.quit()

# Extract and store article titles and URLs
articles = []
seen = set()

errors = ['Sign in to your CoinDesk account',
          'View price details',
          'Sign up for a free CoinDesk account',
          'Markets',
          'Finance',
          'Opinion',
          'Policy',
          'CoinDesk Indices',
          'Tech',
          'Crypto Daybook Americas',
          'Web3',
          'Consensus Toronto 2025 Coverage',
          'CoinDesk homepage'
]

for a in soup.find_all('a', href=True, title=True):
    href = urljoin(url, a['href'])
    title = a['title'].strip()

    if href not in seen and title not in errors:
        seen.add(href)
        articles.append({
            "title": title,
            "url": href
        })

# Save to News.json
with open("links1.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(articles)} articles to News.json")