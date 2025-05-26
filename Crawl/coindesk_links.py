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
options.add_argument("--headless")  # Enable headless mode
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

# Initialize WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Open the CoinDesk Markets page
url = "https://www.coindesk.com/markets"
driver.get(url)

# ✅ Handle cookie consent banner
try:
    accept_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'I agree')]"))
    )
    accept_button.click()
    print("✅ Accepted cookies.")
except:
    print("⚠️ No cookie banner found or already accepted.")

# Scroll settings
scroll_pause_time = 2
max_scrolls = 1000
scroll_count = 0

# Track seen articles
articles_seen = set()

cnt = 0

while cnt < 2000:
    try:
        # Count current <a> tags with href and title using Selenium
        current_count = len(driver.find_elements(By.XPATH, "//a[@href and @title]"))

        # Scroll to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)

        # Try clicking the "More stories" button if present
        try:
            button = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.XPATH, "//button[contains(text(), 'More stories')]"))
            )
            ActionChains(driver).move_to_element(button).click().perform()
            print("🟢 Clicked 'More stories' button.")
            cnt += 1
            time.sleep(scroll_pause_time)
        except:
            print("⚠️ No visible 'More stories' button — relying on scroll.")

        # Wait for more articles to load
        WebDriverWait(driver, 30).until(
            lambda d: len(d.find_elements(By.XPATH, "//a[@href and @title]")) > current_count
        )

        scroll_count += 1
        if scroll_count >= max_scrolls:
            print("🛑 Reached maximum scroll limit.")
            break

    except Exception as e:
        print("⚠️ No new articles found or timeout occurred.")
        break

# Final parse with BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'html.parser')
driver.quit()

# Filter and collect article links
articles = []
seen_urls = set()
excluded_titles = {
    'Sign in to your CoinDesk account',
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
}

for a in soup.find_all('a', href=True, title=True):
    href = urljoin(url, a['href'])
    title = a['title'].strip()

    if href not in seen_urls and title not in excluded_titles:
        seen_urls.add(href)
        articles.append({
            "title": title,
            "url": href
        })

# Save to JSON file
with open("links1.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(articles)} articles to links1.json")