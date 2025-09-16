from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import time
import sys
import random

def save_results(filename, data):
    """Save results to CSV."""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "link", "year", "citations"])
        writer.writeheader()
        writer.writerows(data)
    print(f"✅ Saved {len(data)} results to {filename}")

# List of some common User-Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
    " Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)"
    " Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"
    " Chrome/115.0.0.0 Safari/537.36",
    # Add more user agents here if you want
]

# Choose a random User-Agent for this session
random_ua = random.choice(USER_AGENTS)
print(f"Using User-Agent: {random_ua}")

options = webdriver.ChromeOptions()
options.add_argument(f"user-agent={random_ua}")

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 100)
results = []

for start in range(0, 100, 10):
    url = (
        f"""
        https://scholar.google.com/scholar?start={start}&q=%22deep+reinforcement+learning%22+AND+(%22Dow+Jones+Industrial+Average%22+OR+%22DJIA%22+OR+%22Dow+30%22+OR+%22DOW30%22)&hl=en&as_sdt=0,5&as_ylo=2021
        """
    )
    print(f"🔍 Fetching page starting at {start}...")
    driver.get(url)

    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#gs_res_ccl_mid .gs_r")))

    entries = driver.find_elements(By.CSS_SELECTOR, "#gs_res_ccl_mid .gs_r")

    import re

    for entry in entries:
        title_el = entry.find_element(By.CSS_SELECTOR, ".gs_rt")
        link_el = title_el.find_element(By.TAG_NAME, "a") if title_el.find_elements(By.TAG_NAME, "a") else None

        title = link_el.text if link_el else title_el.text
        link = link_el.get_attribute("href") if link_el else None
        authors_text = entry.find_element(By.CSS_SELECTOR, ".gs_a").text if entry.find_elements(By.CSS_SELECTOR, ".gs_a") else ""
        
        # Extract year using regex (looks for 4-digit number starting with 19 or 20)
        year_match = re.search(r"\b(19|20)\d{2}\b", authors_text)
        year = year_match.group(0) if year_match else ""

        cited_el = [a for a in entry.find_elements(By.TAG_NAME, "a") if "Cited by" in a.text]
        citations = cited_el[0].text if cited_el else "Cited by 0"


        if int(citations.split()[-1]) >= 2:
            results.append({
                "title": title,
                "link": link,
                "year": year,
                "citations": citations
            })

    print(f"✅ Found {len(results)} results so far.")
    save_results("scholar_results.csv", results)
    time.sleep(random.uniform(1, 3))  # random delay between 1-3 seconds
