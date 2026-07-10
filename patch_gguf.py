import sys
from gguf import GGUFReader, GGUFWriter

def patch_gguf(input_path, output_path):
    print(f"Reading {input_path}")
    reader = GGUFReader(input_path)
    
    writer = GGUFWriter(path=output_path, arch=reader.fields["general.architecture"].parts[-1].decode('utf-8'))
    
    # Copy all KV pairs
    for key, field in reader.fields.items():
        if key == "general.architecture":
            continue
        
        val = field.parts[-1]
        
        if key == "qwen35.block_count":
            # The value is a uint32, in field.parts[-1] it's a list or single value
            old_val = val[0] if isinstance(val, list) else val
            print(f"Changing {key} from {old_val} to 64")
            writer.add_uint32(key, 64)
            continue
            
        # Copy the original key-value
        # Actually GGUFWriter handles adding fields differently, maybe we can just modify the raw reader
        pass

# It's easier to modify the field directly if we can, or just use GGUFWriter
# Actually, let's use the gguf-set-metadata tool if it exists, or just read/write the binary file
