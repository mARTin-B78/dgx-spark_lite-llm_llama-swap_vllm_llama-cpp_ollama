import os, json, glob

data = []
for file in glob.glob("test-results/benchmarks/*_20260704_*.json"):
    if "Qwen" not in file and "AEON" not in file and "Prisma" not in file: continue
    try:
        with open(file, "r") as f:
            j = json.load(f)
            model = j.get("model", os.path.basename(file).split("_2026")[0])
            speed = j.get("results", {}).get("tg128", {}).get("t/s", "N/A")
            quality = "N/A"
            if "quality" in j:
                quality = j["quality"].get("score", "N/A")
            data.append((model, speed, quality))
    except Exception as e:
        print(f"Error parsing {file}: {e}")

print("| Model | Speed (tok/s) | Quality (Score) |")
print("|---|---|---|")
for model, speed, quality in sorted(data, key=lambda x: str(x[0])):
    if isinstance(speed, float): speed = round(speed, 2)
    print(f"| {model} | {speed} | {quality} |")
