import json
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# -------------------------
# Setup
# -------------------------
options = Options()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--start-maximized")
# options.add_argument("--headless")

driver = webdriver.Chrome(options=options)

output_file = "JPM.json"
all_data = []

# If file exists, load previous progress
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        try:
            all_data = json.load(f)
            print(f"Loaded {len(all_data)} existing records.")
        except json.JSONDecodeError:
            print("Existing JSON file is empty or broken, starting fresh.")

sources = [
    "wsj.com", "barrons.com", "marketwatch.com", "ft.com", "reuters.com",
    "bloomberg.com", "cnbc.com", "investors.com", "seekingalpha.com",
    "forbes.com", "businessinsider.com", "usatoday.com", "nytimes.com",
    "wired.com", "techcrunch.com", "theverge.com", "theguardian.com",
    "bbc.com", "economist.com"
]

"""
sources = [
    "wsj.com",              # Wall Street Journal
    "barrons.com",          # Barron's
    "marketwatch.com",      # MarketWatch
    "ft.com",               # Financial Times
    "reuters.com",          # Reuters
    "bloomberg.com",        # Bloomberg
    "cnbc.com",             # CNBC
    "investors.com",        # Investor's Business Daily
    "seekingalpha.com",     # Analyst/Investor insights
    "forbes.com",           # Forbes business coverage
    "businessinsider.com",  # Business Insider
    "usatoday.com",         # USA Today (general finance)
    "nytimes.com",          # New York Times
    "theguardian.com",      # Guardian (energy & climate news)
    "bbc.com",              # BBC Business
    "economist.com",        # The Economist
    "finance.yahoo.com",    # Yahoo Finance
    "fool.com",             # The Motley Fool
    "zacks.com",            # Zacks Investment Research
    "marketbeat.com",       # MarketBeat news + analyst ratings
    "oilprice.com",         # OilPrice (energy/commodities specific)
    "rigzone.com",          # Rigzone (oil & gas industry)
    "energyintel.com"       # Energy Intelligence
]


sources = [
    # Core financial media
    "wsj.com",              # Wall Street Journal
    "barrons.com",          # Barron's
    "marketwatch.com",      # MarketWatch
    "ft.com",               # Financial Times
    "reuters.com",          # Reuters
    "bloomberg.com",        # Bloomberg
    "cnbc.com",             # CNBC
    "investors.com",        # Investor's Business Daily
    "seekingalpha.com",     # Analyst/Investor insights
    "forbes.com",           # Forbes
    "businessinsider.com",  # Business Insider
    "usatoday.com",         # USA Today (business desk only)
    "nytimes.com",          # NYT business coverage
    "theguardian.com/business", # Guardian business
    "bbc.com/news/business",    # BBC business
    "economist.com",        # The Economist

    # Stock analysis & earnings
    "finance.yahoo.com",    # Yahoo Finance
    "fool.com",             # Motley Fool
    "zacks.com",            # Zacks Investment Research
    "marketbeat.com",       # MarketBeat

    # Industry/defense sources (financial impact via contracts, regulations, safety)
    "aviationweek.com",     # Aviation Week — industry, but heavily finance-relevant
    "breakingdefense.com",  # Defense contracting & aerospace industry
    "flightglobal.com"      # FlightGlobal — commercial aircraft orders, airline deals
]
"""

# Build the sources string in parentheses
sources_query = " OR ".join([f"site:{s}" for s in sources])
#index_query = "(AAPL OR Apple OR \"Apple Inc.\")"
index_query = "(JPM OR JPMorgan OR \"JPMorgan Chase\")" 
# index_query = "(BA OR Boeing)"

# -------------------------
# Crawling
# -------------------------
for year in range(2018, 2019):
    la = 0
    for month in range(la, 13):  # avoid month=0
        start = 0            
        while start < 50:
            print(f"[INFO] Year={year} Month={month} PageStart={start} | Total so far: {len(all_data)}")

            url = (
                f"https://www.google.com/search?q={index_query}+{sources_query}"
                f"&tbs=cdr:1,cd_min:{month}/1/{year},cd_max:{month}/31/{year}"
                f"&tbm=nws&hl=en&gl=us&start={start}"
            )
            driver.get(url)
            driver.implicitly_wait(2)

            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_all_elements_located((By.XPATH, '//*[@id="rso"]//a'))
                )
            except TimeoutException:
                print(f"[STOP] No more results for {year}-{month} start={start}")
                break

            articles = driver.find_elements(By.XPATH, '//*[@id="rso"]//a')
            page_data = []

            for el in articles:
                try:
                    link = el.get_attribute("href")
                    if not link or any(d["link"] == link for d in all_data):
                        continue

                    headline = el.find_element(By.XPATH, './/div[@role="heading"]').text.strip()

                    try:
                        source_date = el.find_element(By.XPATH, './/div[4]').text.strip()
                    except:
                        source_date = ""

                    try:
                        summary = el.find_element(By.XPATH, './/div[3]').text.strip()
                    except:
                        summary = ""

                    page_data.append({
                        "year": year,
                        "month": month,
                        "headline": headline,
                        "link": link,
                        "source": source_date,
                        "summary": summary
                    })
                except:
                    continue

            if not page_data:
                print("[WARN] No new articles found, stopping month.")
                break

            all_data.extend(page_data)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            print(f"[SAVE] {len(page_data)} items saved. Total now: {len(all_data)}")

            start += len(page_data)  # dynamic increment

driver.quit()
print(f"Finished. Total records: {len(all_data)}")
