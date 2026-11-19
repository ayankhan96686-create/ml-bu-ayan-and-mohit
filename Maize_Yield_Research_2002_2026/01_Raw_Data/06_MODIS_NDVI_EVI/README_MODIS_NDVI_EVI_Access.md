# MODIS MOD13Q1 (NDVI & EVI) Access Guide (NASA Earthdata / LP DAAC)

## Overview
NASA's Moderate Resolution Imaging Spectroradiometer (MODIS) aboard Terra provides the MOD13Q1 Version 6.1 product:
- **Spatial Resolution**: 250 meters
- **Temporal Resolution**: 16-day composite
- **Temporal Coverage**: February 2000 to present (covers entire 2002–2026 study timeframe)
- **Key Layers**:
  - `250m_16_days_NDVI` (Normalized Difference Vegetation Index, scale factor 0.0001)
  - `250m_16_days_EVI` (Enhanced Vegetation Index, scale factor 0.0001)
  - `250m_16_days_VI_Quality` (Per-pixel quality reliability bitmask)
  - Surface Reflectance bands: Red (Band 1), NIR (Band 2), Blue (Band 3), Mid-IR (Band 7)

## Access Requirements & Blocker Documentation
1. **NASA Earthdata Account**: Requires free registration at https://urs.earthdata.nasa.gov/
2. **Authorization**: Authorize "LP DAAC Data Pool" and "AppEEARS" under Applications in user profile.
3. **Authentication Token**:
   - Save credentials in `~/.netrc` (Windows: `%USERPROFILE%\_netrc`):
     ```text
     machine urs.earthdata.nasa.gov login <YOUR_USERNAME> password <YOUR_PASSWORD>
     ```
4. **Automated Subsetting via AppEEARS API**:
   - NASA AppEEARS (Application for Extracting and Exploring Analysis Ready Samples) enables extracting county-level shapefile or bounding box statistics without downloading massive raw sinusoidal HDF tiles (H10V04, H11V04, H11V05, H12V04, H12V05).
   - API Portal: https://appeears.earthdatacloud.nasa.gov/api/

## Python AppEEARS Submission Script Template
```python
import requests
import json

# Log in to AppEEARS
login_url = 'https://appeears.earthdatacloud.nasa.gov/api/login'
response = requests.post(login_url, auth=('USER', 'PASSWORD'))
token = response.json()['token']

# Submit area task with study boundary GeoJSON
headers = {'Authorization': f'Bearer {token}'}
task_payload = {
    'task_type': 'area',
    'task_name': 'CornBelt_MOD13Q1_2002_2026',
    'params': {
        'dates': [{'startDate': '04-01-2002', 'endDate': '11-01-2026'}],
        'layers': [
            {'product': 'MOD13Q1.061', 'layer': '_250m_16_days_NDVI'},
            {'product': 'MOD13Q1.061', 'layer': '_250m_16_days_EVI'},
            {'product': 'MOD13Q1.061', 'layer': '_250m_16_days_VI_Quality'}
        ],
        'output': {'format': {'type': 'geotiff'}, 'projection': 'geographic'}
    }
}
# requests.post('https://appeears.earthdatacloud.nasa.gov/api/task', headers=headers, json=task_payload)
```

