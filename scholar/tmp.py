# Read the two CSV files
file1 = "/home/influx/Desktop/FinLLMRL/scholar/DOW30/sorted_articles.csv"
file2 = "/home/influx/Desktop/FinLLMRL/scholar/total/sorted_articles.csv"

output_file = "merged_unique.csv"
lines_seen = set()
data = []

# Read both files line by line
for filename in [file1, file2]:
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0:  # skip header for now
                header = line
                continue
            if line not in lines_seen:
                data.append(line)
                lines_seen.add(line)

# Function to extract citation number
def get_citations(line):
    parts = line.rsplit(",", 1)  # get last part
    c = parts[-1].replace("Cited by ", "").strip()
    try:
        return int(c)
    except:
        return 0

# Sort by citations descending
data.sort(key=get_citations, reverse=True)

# Save to file with header
with open(output_file, "w", encoding="utf-8") as f:
    f.write(header + "\n")
    for line in data:
        f.write(line + "\n")

print("✅ Done. Saved as", output_file)
