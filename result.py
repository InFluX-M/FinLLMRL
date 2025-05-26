import json
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

# --- Load and aggregate impact scores ---
with open("result.json", "r") as f:
    data = [json.loads(line) for line in f]


impact_by_date = defaultdict(float)
for entry in data:
    date_str = entry["date"]
    if not date_str:
        continue
    score = float(entry["impact_score"])
    impact_by_date[date_str] += score

# --- Manually entered BTC closing prices ---
btc_price_raw = """
May 20, 2025\t106,165.59
May 18, 2025\t106,446.01
May 17, 2025\t103,191.09
May 16, 2025\t103,489.29
May 15, 2025\t103,744.64
May 14, 2025\t103,539.41
May 13, 2025\t104,169.81
May 12, 2025\t102,812.95
May 11, 2025\t104,106.36
May 10, 2025\t104,696.33
May 9, 2025\t102,970.85
May 8, 2025\t103,241.46
May 7, 2025\t97,032.32
May 6, 2025\t96,802.48
May 5, 2025\t94,748.05
May 4, 2025\t94,315.98
May 3, 2025\t95,891.80
May 2, 2025\t96,910.07
"""

btc_price_map = {}
for line in btc_price_raw.strip().splitlines():
    date_str, price_str = line.split('\t')
    date_obj = datetime.datetime.strptime(date_str.strip(), "%b %d, %Y")
    btc_price_map[date_obj.strftime("%Y-%m-%d")] = float(price_str.replace(",", ""))

# --- Prepare aligned data ---
sorted_items = sorted(impact_by_date.items(), key=lambda x: x[0])
dates = [datetime.datetime.strptime(date, "%Y-%m-%d") for date, _ in sorted_items]
scores = [score for _, score in sorted_items]
btc_prices = [btc_price_map.get(date.strftime("%Y-%m-%d"), None) for date in dates]

# --- Plotting ---
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot impact score (left y-axis)
ax1.set_xlabel("Date")
ax1.set_ylabel("Impact Score", color="blue")
ax1.plot(dates, scores, marker='o', linestyle='-', color="blue", label="Impact Score")
ax1.tick_params(axis='y', labelcolor="blue")

# Plot BTC price (right y-axis)
ax2 = ax1.twinx()
ax2.set_ylabel("BTC Price (USD)", color="green")
ax2.plot(dates, btc_prices, marker='s', linestyle='--', color="green", label="BTC Price")
ax2.tick_params(axis='y', labelcolor="green")

plt.title("Impact Score vs BTC Price")
plt.grid(True)
plt.xticks(rotation=45)
fig.tight_layout()
plt.show(block=True)
