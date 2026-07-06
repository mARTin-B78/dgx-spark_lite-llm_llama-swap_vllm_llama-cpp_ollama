import glob, os

files = glob.glob("test-results/benchmarks/report_20260704_0[4-7]*.txt")

print("| Model | Read (pp) | Write (tg) | Deep ctx | Quality |")
print("|---|---|---|---|---|")

for f in sorted(files):
    with open(f, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Qwen" in line or "Nemotron" in line:
                if "FAIL" not in line and "tok/s" not in line and "degradation" not in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        model = parts[0]
                        read = parts[2]
                        write = parts[3]
                        deep = parts[5] if "%" in parts[5] else parts[6] if len(parts) > 6 else ""
                        quality = parts[-1] if "/" in parts[-1] or parts[-1].isdigit() else "N/A"
                        print(f"| {model} | {read} | {write} | {deep} | {quality} |")
