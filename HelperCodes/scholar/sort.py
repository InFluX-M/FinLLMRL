import pandas as pd

# Read the CSV
df = pd.read_csv("scholar_results.csv")

# Extract citation numbers from strings like "Cited by 292"
df["citations"] = df["citations"].astype(str).str.extract(r'(\d+)').astype(int)

# Sort by citations in descending order
df_sorted = df.sort_values(by="citations", ascending=False)

# Save the sorted CSV
df_sorted.to_csv("sorted_articles.csv", index=False)

print("Sorted CSV saved as sorted_articles.csv")
