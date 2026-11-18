import urllib.request
import os
import time

url = "https://nassgeodata.gmu.edu/webservice/nass_data_cache/byfips/CDL_2020_19.tif"
dest = r"E:\ml do not open\Maize_Yield_Research_2002_2026\01_Raw_Data\02_USDA_CDL_Crop_Classification\CDL_2020_19_Iowa.tif"

print(f"Downloading {url} to {dest} using chunked streaming with retries...")

chunk_size = 1024 * 1024 # 1MB chunks
max_retries = 5

for attempt in range(max_retries):
    existing_bytes = os.path.getsize(dest) if os.path.exists(dest) else 0
    headers = {'User-Agent': 'Mozilla/5.0'}
    if existing_bytes > 0:
        headers['Range'] = f"bytes={existing_bytes}-"
        mode = 'ab'
        print(f"Resuming download from byte {existing_bytes}...")
    else:
        mode = 'wb'
        print("Starting new download...")

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            # If 206 Partial Content or 200 OK
            total_expected = resp.headers.get('Content-Length')
            with open(dest, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
        print(f"Download complete: {dest} ({os.path.getsize(dest)} bytes)")
        break
    except Exception as e:
        print(f"Attempt {attempt+1} encountered: {e}. Retrying in 2 seconds...")
        time.sleep(2)

