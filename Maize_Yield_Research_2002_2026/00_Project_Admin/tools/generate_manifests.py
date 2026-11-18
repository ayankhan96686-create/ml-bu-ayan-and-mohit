import os
import hashlib
import csv
from datetime import datetime

base_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026"
manifests_dir = os.path.join(base_dir, "04_Manifests_and_Checksums")
os.makedirs(manifests_dir, exist_ok=True)

checksum_csv = os.path.join(manifests_dir, "SHA256_Checksums.csv")
inventory_csv = os.path.join(manifests_dir, "File_Inventory.csv")
duplicate_csv = os.path.join(manifests_dir, "Duplicate_File_Report.csv")

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

files_info = []
hash_map = {}

print("Scanning complete repository and updating SHA256 checksums...")

all_dirs = ["00_Project_Admin", "01_Raw_Data", "02_Extracted_Data", "03_Processed_Data", "05_Reports"]

for td in all_dirs:
    full_td = os.path.join(base_dir, td)
    if not os.path.exists(full_td):
        continue
    for root, dirs, files in os.walk(full_td):
        for fn in files:
            fpath = os.path.join(root, fn)
            rel_path = os.path.relpath(fpath, base_dir)
            size = os.path.getsize(fpath)
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime('%Y-%m-%d %H:%M:%S')
            ext = os.path.splitext(fn)[1].lower()
            
            sha = get_sha256(fpath)
            
            files_info.append({
                'Relative_Path': rel_path,
                'File_Name': fn,
                'File_Extension': ext,
                'Size_Bytes': size,
                'Size_MB': round(size / (1024 * 1024), 3),
                'Last_Modified': mtime,
                'SHA256': sha
            })
            
            if sha not in hash_map:
                hash_map[sha] = []
            hash_map[sha].append(rel_path)

print(f"Total files cataloged: {len(files_info)}")

with open(checksum_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['Relative_Path', 'File_Name', 'Size_Bytes', 'SHA256'])
    writer.writeheader()
    for item in sorted(files_info, key=lambda x: x['Relative_Path']):
        writer.writerow({
            'Relative_Path': item['Relative_Path'],
            'File_Name': item['File_Name'],
            'Size_Bytes': item['Size_Bytes'],
            'SHA256': item['SHA256']
        })

with open(inventory_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['Relative_Path', 'File_Name', 'File_Extension', 'Size_Bytes', 'Size_MB', 'Last_Modified', 'SHA256'])
    writer.writeheader()
    for item in sorted(files_info, key=lambda x: x['Relative_Path']):
        writer.writerow(item)

duplicates = []
for sha, paths in hash_map.items():
    if len(paths) > 1:
        duplicates.append({
            'SHA256': sha,
            'Duplicate_Count': len(paths),
            'File_Paths': " | ".join(paths)
        })

with open(duplicate_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['SHA256', 'Duplicate_Count', 'File_Paths'])
    writer.writeheader()
    for item in duplicates:
        writer.writerow(item)

print(f"Complete manifests generated:")
print(f"  Checksums: {checksum_csv}")
print(f"  Inventory: {inventory_csv}")
print(f"  Duplicate Report: {duplicate_csv} ({len(duplicates)} duplicate sets found)")