import os

base_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026"

def get_dir_stats(folder):
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(folder):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
                file_count += 1
    return file_count, total_size

print(f"{'Category / Dataset':<50} {'Files':>8} {'Size (MB)':>12}")
print("-" * 72)

raw_base = os.path.join(base_dir, "01_Raw_Data")
for sub in sorted(os.listdir(raw_base)):
    sp = os.path.join(raw_base, sub)
    if os.path.isdir(sp):
        fc, sz = get_dir_stats(sp)
        print(f"01_Raw_Data/{sub:<35} {fc:>8} {round(sz/(1024*1024), 2):>12}")

print("-" * 72)
ext_base = os.path.join(base_dir, "02_Extracted_Data")
for sub in sorted(os.listdir(ext_base)):
    sp = os.path.join(ext_base, sub)
    if os.path.isdir(sp):
        fc, sz = get_dir_stats(sp)
        print(f"02_Extracted_Data/{sub:<31} {fc:>8} {round(sz/(1024*1024), 2):>12}")

print("=" * 72)
all_fc, all_sz = get_dir_stats(base_dir)
print(f"{'TOTAL PROJECT STORAGE':<50} {all_fc:>8} {round(all_sz/(1024*1024), 2):>12} MB ({round(all_sz/(1024*1024*1024), 3)} GB)")

