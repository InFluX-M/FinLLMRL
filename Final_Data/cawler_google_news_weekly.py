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

# index_query = "(AAPL OR Apple OR \"Apple Inc.\")"
# index_query = "(JPM OR JPMorgan OR \"JPMorgan Chase\")" 
# index_query = "(BA OR Boeing)"

from datetime import datetime, timedelta
import re

def parse_source_date(source_date: str, today: datetime = None):
    if today is None:
        today = datetime.today()
    
    source_date = source_date.strip()

    # Try absolute date first
    try:
        return datetime.strptime(source_date, "%b %d, %Y").date()
    except ValueError:
        pass
    
    # Relative date pattern
    match = re.match(r"(\d+)\s+(day|week|hour|month|year)s?\s+ago", source_date.lower())
    if match:
        value, unit = int(match.group(1)), match.group(2)

        if unit == "day":
            return (today - timedelta(days=value)).date()
        elif unit == "week":
            return (today - timedelta(weeks=value)).date()
        elif unit == "hour":
            return (today - timedelta(hours=value)).date()
        elif unit == "month":
            return (today - timedelta(days=30 * value)).date()  # approx
        elif unit == "year":
            return (today - timedelta(days=365 * value)).date()  # approx
    
    raise ValueError(f"Unrecognized date format: {source_date}")

def crawl(index_query, year, month, day, length, out_file, mx_len):
    all_data = []
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            try:
                all_data = json.load(f)
                print(f"Loaded existing records.")
            except json.JSONDecodeError:
                print("Existing JSON file is empty or broken, starting fresh.")

    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    # options.add_argument("--headless")

    driver = webdriver.Chrome(options=options)

    start = 0            
    while start < mx_len:
        url = (
            f"https://www.google.com/search?q={index_query}"
            f"&tbs=cdr:1,cd_min:{month}/{day}/{year},cd_max:{month}/{day + length - 1}/{year}"
            f"&tbm=nws&hl=en&gl=us&start={start}"
        )
        
        driver.get(url)
        driver.implicitly_wait(2)

        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_all_elements_located((By.XPATH, '//*[@id="rso"]//a'))
            )
        except TimeoutException:
            print(f"[STOP] No more results")
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

                dt = parse_source_date(source_date, today=datetime.now())
                try:
                    dt = parse_source_date(source_date, today=datetime.now())
                except Exception:
                    try:
                        dt = parse_source_date(summary, today=datetime.now())
                    except Exception:
                        try:
                            dt = datetime.strptime(source_date, "%b %Y").date().replace(day=1)
                        except Exception:
                            try:
                                dt = datetime.strptime(source_date, "%Y").date().replace(month=1, day=1)
                            except Exception:
                                continue

                page_data.append({
                    "headline": headline,
                    "link": link,
                    "summary": summary,
                    "date": dt.isoformat()
                })
            except:
                continue

        if not page_data:
            print("[WARN] No new articles found.")
            break

        all_data.extend(page_data)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] items saved")

        start += len(page_data)  # dynamic increment

    driver.quit()
    print(f"Finished. Total records: {len(all_data)}")


# Example usage
if __name__ == "__main__":
    # index_query = "(AAPL)"
    # index_query = "(JPM OR \"JPMorgan Chase\")" 
    # index_query = "(BA OR Boeing)"
    index_query = "(GS OR OR \"Goldman Sachs\")"
    stock = "GS"
    crawl(index_query, 2025, 8, 15, 9, f"new_{stock}.json", 50)
    crawl(index_query, 2025, 8, 23, 9, f"new_{stock}.json", 50)
    crawl(index_query, 2025, 9, 1, 7, f"new_{stock}.json", 50)
