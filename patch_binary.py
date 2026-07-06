import os

file_path = "/home/sparky/LLMs/ollama/Alibaba/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-NVFP4-GGUF/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-NVFP4-Q8_0.gguf"

key_str = b"qwen35.block_count"
key_len_bytes = len(key_str).to_bytes(8, 'little')
value_type = (4).to_bytes(4, 'little') # uint32 is type 4
old_val = (65).to_bytes(4, 'little')
new_val = (64).to_bytes(4, 'little')

search_pattern = key_len_bytes + key_str + value_type + old_val
replace_pattern = key_len_bytes + key_str + value_type + new_val

with open(file_path, 'r+b') as f:
    # Read first 1MB, header is well within this
    data = f.read(1024 * 1024)
    idx = data.find(search_pattern)
    if idx != -1:
        print(f"Found pattern at offset {idx}")
        f.seek(idx + len(search_pattern) - 4)
        f.write(new_val)
        print("Patched successfully!")
    else:
        print("Pattern not found! Searching without key_len_bytes...")
        # Maybe key length is encoded differently (uint32?)
        # Let's search just the string + type + val
        pattern2 = key_str + value_type + old_val
        idx2 = data.find(pattern2)
        if idx2 != -1:
             print(f"Found pattern2 at offset {idx2}")
             f.seek(idx2 + len(pattern2) - 4)
             f.write(new_val)
             print("Patched successfully!")
        else:
             print("Pattern2 not found either.")
