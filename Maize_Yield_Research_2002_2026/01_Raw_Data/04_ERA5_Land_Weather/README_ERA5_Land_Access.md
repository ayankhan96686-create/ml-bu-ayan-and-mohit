# ERA5-Land Climate Data Access Guide (ECMWF / Copernicus CDS)

## Overview
ERA5-Land is a global atmospheric reanalysis produced by the European Centre for Medium-Range Weather Forecasts (ECMWF) providing hourly, daily, and monthly climate variables at a spatial resolution of 9 km (~0.1 degrees) from 1950 to present.

## Key Agricultural Variables
- `2m_temperature` (K)
- `2m_dewpoint_temperature` (K)
- `total_precipitation` (m)
- `surface_solar_radiation_downwards` (J/m^2)
- `volumetric_soil_water_layer_1` (0-7 cm) (m^3/m^3)
- `volumetric_soil_water_layer_2` (7-28 cm) (m^3/m^3)
- `10m_u_component_of_wind` & `10m_v_component_of_wind` (m/s)

## Access Requirements & Blocker Documentation
1. **Account Registration**: Free user account on the Copernicus Climate Data Store (CDS): https://cds.climate.copernicus.eu/
2. **Accept Terms of Use**: Log in to CDS and accept the Copernicus Licence for ERA5-Land.
3. **API Key Generation**: Obtain Personal Access Token (PAT) from user profile.
4. **Configuration File**: Save credentials in `~/.cdsapirc`:
   ```text
   url: https://cds.climate.copernicus.eu/api
   key: <YOUR-PERSONAL-ACCESS-TOKEN>
   ```
5. **Python Client**: Install `cdsapi` (`pip install cdsapi`).

## Regional Bounding Box for Study Area (US Corn Belt)
To avoid downloading global NetCDF/GRIB rasters, query the regional sub-bounding box:
- North: 49.5° N
- West: -104.5° W
- South: 35.5° N
- East: -80.0° W

## Python Extraction Script Template
```python
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-land-monthly-means',
    {
        'format': 'netcdf',
        'variable': [
            '2m_temperature', 'total_precipitation',
            'surface_solar_radiation_downwards',
            'volumetric_soil_water_layer_1',
        ],
        'year': [str(y) for y in range(2002, 2027)],
        'month': [f"{m:02d}" for m in range(1, 13)],
        'time': '00:00',
        'area': [49.5, -104.5, 35.5, -80.0], # North, West, South, East
    },
    'ERA5_Land_Monthly_CornBelt_2002_2026.nc'
)
```

