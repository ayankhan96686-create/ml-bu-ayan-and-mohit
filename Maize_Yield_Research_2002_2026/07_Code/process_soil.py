import os
import urllib.request
import struct
import io
import pandas as pd
import numpy as np
from PIL import Image

base_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026"
soil_dir = os.path.join(base_dir, "01_Raw_Data", "11_Soil_SoilGrids")
regional_raster_dir = os.path.join(soil_dir, "regional_rasters")
os.makedirs(regional_raster_dir, exist_ok=True)

counties_ref_csv = os.path.join(base_dir, "00_Project_Admin", "Study_Area_Counties_FIPS.csv")
county_dbf_path = os.path.join(base_dir, "02_Extracted_Data", "14_County_and_State_Boundaries", "tl_2024_us_county", "tl_2024_us_county.dbf")
out_dir = os.path.join(base_dir, "03_Cleaned_Data")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "SoilGrids_Cleaned.csv")

print("Executing Phase 4: ISRIC SoilGrids 2.0 Regional Processing & Cleaning...")

# 1. Load official 814 study counties
df_counties = pd.read_csv(counties_ref_csv, dtype=str)
valid_fips = set(df_counties['Full_FIPS'].str.strip())
fips_to_meta = df_counties.set_index('Full_FIPS')[['State_Name', 'County_Name']].to_dict('index')

# 2. Parse county centroids from DBF
county_coords = {}
with open(county_dbf_path, 'rb') as f:
    header = f.read(32)
    num_records, header_len, record_len = struct.unpack('<IHH', header[4:12])
    fields = []
    while True:
        field_desc = f.read(32)
        if field_desc[0] == 0x0D:
            break
        name = field_desc[:11].replace(b'\x00', b'').decode('ascii')
        typ = chr(field_desc[11])
        length = field_desc[16]
        fields.append((name, typ, length))
        
    f.seek(header_len)
    for _ in range(num_records):
        rec_bytes = f.read(record_len)
        if len(rec_bytes) < record_len:
            break
        offset = 1
        rec_dict = {}
        for fname, ftyp, flen in fields:
            val = rec_bytes[offset:offset+flen].decode('latin1', errors='ignore').strip()
            rec_dict[fname] = val
            offset += flen
            
        st = rec_dict.get('STATEFP', '').zfill(2)
        cty = rec_dict.get('COUNTYFP', '').zfill(3)
        geoid = st + cty
        if geoid in valid_fips:
            try:
                lat = float(rec_dict.get('INTPTLAT', '0'))
                lon = float(rec_dict.get('INTPTLON', '0'))
                county_coords[geoid] = (lat, lon)
            except:
                pass

print(f"Loaded {len(county_coords)} study county centroids.")

# Regional Grid Bounding Box for US Corn Belt:
# Lon: -104.5 to -80.0, Lat: 36.0 to 49.5
MIN_LON = -104.5
MAX_LON = -80.0
MIN_LAT = 36.0
MAX_LAT = 49.5
GRID_X = 1000 # columns (width)
GRID_Y = 600  # rows (height)

# Coordinate conversion functions
# lon, lat to pixel col, row
def coord_to_pixel(lat, lon):
    c = int((lon - MIN_LON) / (MAX_LON - MIN_LON) * (GRID_X - 1))
    r = int((MAX_LAT - lat) / (MAX_LAT - MIN_LAT) * (GRID_Y - 1))
    return max(0, min(GRID_Y - 1, r)), max(0, min(GRID_X - 1, c))

properties = ['clay', 'sand', 'silt', 'soc', 'phh2o', 'bdod']
depths = ['0-5cm', '5-15cm', '15-30cm']

# Scale factors according to official SoilGrids specifications
# clay, sand, silt: g/kg -> divide by 10 for %
# soc: dg/kg -> divide by 100 for %
# phh2o: pH*10 -> divide by 10 for standard pH
# bdod: cg/cm^3 -> divide by 100 for g/cm^3
prop_scales = {
    'clay': 0.1,
    'sand': 0.1,
    'silt': 0.1,
    'soc': 0.01,
    'phh2o': 0.1,
    'bdod': 0.01
}

raster_grids = {}

for prop in properties:
    for d in depths:
        cov_id = f"{prop}_{d}_mean"
        cached_tif = os.path.join(regional_raster_dir, f"{cov_id}_CornBelt.tif")
        
        if not os.path.exists(cached_tif) or os.path.getsize(cached_tif) < 5000:
            print(f"Fetching WCS regional GeoTIFF for {cov_id} from ISRIC...")
            url = f"https://maps.isric.org/mapserv?map=/map/{prop}.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID={cov_id}&SUBSET=X({MIN_LON},{MAX_LON})&SUBSET=Y({MIN_LAT},{MAX_LAT})&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326&SCALESIZE=x({GRID_X}),y({GRID_Y})&FORMAT=image/tiff"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp, open(cached_tif, 'wb') as f:
                    f.write(resp.read())
                print(f"  Saved {os.path.basename(cached_tif)} ({os.path.getsize(cached_tif)} bytes)")
            except Exception as e:
                print(f"  Error fetching {cov_id}: {e}")
                
        if os.path.exists(cached_tif):
            im = Image.open(cached_tif)
            raster_grids[cov_id] = np.array(im, dtype=np.float32)
        else:
            raster_grids[cov_id] = None

# Extract county soil statistics
county_soil_rows = []

for fips in sorted(county_coords.keys()):
    lat, lon = county_coords[fips]
    r, c = coord_to_pixel(lat, lon)
    st_name = fips_to_meta[fips]['State_Name']
    cty_name = fips_to_meta[fips]['County_Name']
    
    row = {
        'county_fips': fips,
        'state': st_name,
        'county': cty_name,
        'intptlat': round(lat, 5),
        'intptlon': round(lon, 5)
    }
    
    for prop in properties:
        scale = prop_scales[prop]
        for d in depths:
            cov_id = f"{prop}_{d}_mean"
            clean_col = f"{prop}_{d.replace('-', '_')}"
            grid = raster_grids.get(cov_id)
            if grid is not None:
                # 3x3 window around centroid
                r_min, r_max = max(0, r-1), min(GRID_Y, r+2)
                c_min, c_max = max(0, c-1), min(GRID_X, c+2)
                sub = grid[r_min:r_max, c_min:c_max]
                valid = sub[(sub > 0) & (sub < 30000)]
                if len(valid) > 0:
                    val = float(valid.mean()) * scale
                    row[clean_col] = round(val, 2)
                else:
                    row[clean_col] = np.nan
            else:
                row[clean_col] = np.nan
                
    county_soil_rows.append(row)

df_soil_clean = pd.DataFrame(county_soil_rows).sort_values('county_fips').reset_index(drop=True)
df_soil_clean.to_csv(out_file, index=False)

print(f"Saved clean SoilGrids dataset to: {out_file}")
print(f"Total counties: {len(df_soil_clean):,}")
print("Sample clean soil statistics:")
cols_to_show = ['clay_0_5cm', 'sand_0_5cm', 'silt_0_5cm', 'soc_0_5cm', 'phh2o_0_5cm', 'bdod_0_5cm']
print(df_soil_clean[cols_to_show].describe().round(2))

