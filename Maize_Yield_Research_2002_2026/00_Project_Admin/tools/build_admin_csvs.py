import csv
import os

admin_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\00_Project_Admin"
os.makedirs(admin_dir, exist_ok=True)

# 1. Source_Registry.csv
source_registry_path = os.path.join(admin_dir, "Source_Registry.csv")
sources = [
    {
        "Source_ID": "SRC-01",
        "Source_Name": "USDA NASS Quick Stats",
        "Organization": "United States Department of Agriculture, National Agricultural Statistics Service",
        "Primary_URL": "https://quickstats.nass.usda.gov/",
        "Data_Type": "Observed Crop Yield, Survey Acreage, Crop Progress & Condition, Fertilizer Practices",
        "Access_Method": "Official Web Query & Direct Browser CSV Download (UUID session endpoint)",
        "Auth_Required": "None for browser portal (REST API requires free key)",
        "License": "US Public Domain / Open Government Data",
        "Target_Coverage": "2002-2026 (9 Corn Belt States)"
    },
    {
        "Source_ID": "SRC-02",
        "Source_Name": "USDA NASS Cropland Data Layer (CDL) / CropScape",
        "Organization": "USDA NASS & George Mason University CSISS",
        "Primary_URL": "https://croplandcros.scinet.usda.gov/ & https://nassgeodata.gmu.edu/CropScape/",
        "Data_Type": "Annual Crop Specific Land Cover Raster (30m GeoTIFF)",
        "Access_Method": "CropScape CDLService SOAP/REST API & NASS Releases",
        "Auth_Required": "None (Public Web Service)",
        "License": "US Public Domain",
        "Target_Coverage": "2008-2025 CONUS (2002-2007 Partial Midwest states; 2026 Unpublished)"
    },
    {
        "Source_ID": "SRC-03",
        "Source_Name": "PRISM Climate Group",
        "Organization": "Northwest Alliance for Computational Science and Engineering (NACSE) / Oregon State University",
        "Primary_URL": "https://prism.oregonstate.edu/",
        "Data_Type": "Gridded Daily/Monthly/Annual Climate (ppt, tmin, tmax, tmean) at 4km",
        "Access_Method": "PRISM Web Service API (services.nacse.org/prism/data/get/)",
        "Auth_Required": "None for 4km public datasets",
        "License": "PRISM Terms of Use (Free for academic and non-commercial research)",
        "Target_Coverage": "2002-2025 Final Annual; 2026 Jan-Oct Provisional Monthly"
    },
    {
        "Source_ID": "SRC-04",
        "Source_Name": "NOAA NCEI Climate at a Glance (CAG) / ClimDiv",
        "Organization": "National Oceanic and Atmospheric Administration, National Centers for Environmental Information",
        "Primary_URL": "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/",
        "Data_Type": "County-level Monthly Climate Observations (pcpn, tmpc, tmax, tmin)",
        "Access_Method": "Direct HTTPS Bulk File Transfer",
        "Auth_Required": "None (Public Open Access)",
        "License": "US Public Domain / Open Government Data",
        "Target_Coverage": "1895-2026 (Continuous monthly coverage through October 2026)"
    },
    {
        "Source_ID": "SRC-05",
        "Source_Name": "NASA POWER Agroclimatology",
        "Organization": "NASA Langley Research Center",
        "Primary_URL": "https://power.larc.nasa.gov/",
        "Data_Type": "Solar Radiation (ALLSKY_SFC_SW_DWN), Temperature, Humidity, Precipitation",
        "Access_Method": "NASA POWER REST API",
        "Auth_Required": "None (Public Open Access)",
        "License": "NASA Open Data Policy (CC0 equivalent)",
        "Target_Coverage": "2002-2026 (Daily observations)"
    },
    {
        "Source_ID": "SRC-06",
        "Source_Name": "USDA RMA Summary of Business",
        "Organization": "USDA Risk Management Agency",
        "Primary_URL": "https://www.rma.usda.gov/tools-reports/summary-of-business/state-county-crop-summary-business",
        "Data_Type": "County-level Crop Insurance, Planted/Insured Acres, Liabilities, Indemnities",
        "Access_Method": "Direct HTTPS Bulk ZIP Archive Downloads",
        "Auth_Required": "None (Public Open Access)",
        "License": "US Public Domain / Open Government Data",
        "Target_Coverage": "2002-2026 (Annual state_county_crop flat files)"
    },
    {
        "Source_ID": "SRC-07",
        "Source_Name": "USDA ERS Fertilizer Use and Price",
        "Organization": "USDA Economic Research Service",
        "Primary_URL": "https://www.ers.usda.gov/data-products/fertilizer-use-and-price/",
        "Data_Type": "Crop-specific Fertilizer Application Rates, Nutrient Consumption, Price Indices",
        "Access_Method": "Direct HTTPS Excel Workbook Download",
        "Auth_Required": "None (Public Open Access)",
        "License": "US Public Domain / Open Government Data",
        "Target_Coverage": "Historical through 2024 (multi-year static workbook)"
    },
    {
        "Source_ID": "SRC-08",
        "Source_Name": "US Census Bureau TIGER/Line",
        "Organization": "United States Census Bureau, Department of Commerce",
        "Primary_URL": "https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html",
        "Data_Type": "State & County Geographic Boundary Shapefiles & National FIPS Codes",
        "Access_Method": "Direct HTTPS Archive Downloads",
        "Auth_Required": "None (Public Open Access)",
        "License": "US Public Domain / Open Government Data",
        "Target_Coverage": "2024 Vintage (covering all 814 study area counties)"
    },
    {
        "Source_ID": "SRC-09",
        "Source_Name": "ISRIC SoilGrids",
        "Organization": "ISRIC - World Soil Information",
        "Primary_URL": "https://soilgrids.org/ & https://files.isric.org/soilgrids/latest/data/",
        "Data_Type": "Global 250m Soil Physical & Chemical Properties (clay, sand, silt, soc, phh2o, bdod)",
        "Access_Method": "WebDAV / HTTPS VRT & GeoTIFF Tile Server",
        "Auth_Required": "None (Public Open Access)",
        "License": "Creative Commons Attribution 4.0 International (CC-BY 4.0)",
        "Target_Coverage": "Static pedogenic baseline (Standard depth intervals: 0-5cm, 5-15cm, 15-30cm, etc.)"
    },
    {
        "Source_ID": "SRC-10",
        "Source_Name": "USDA NRCS gSSURGO",
        "Organization": "USDA Natural Resources Conservation Service",
        "Primary_URL": "https://www.nrcs.usda.gov/resources/data-and-reports/gridded-soil-survey-geographic-database-gssurgo",
        "Data_Type": "10m Gridded Soil Survey & Detailed SSURGO Tabular Characteristics (AWC, SOM, pH)",
        "Access_Method": "NRCS Box Cloud Repository (nrcs.app.box.com/v/soils) & Soil Data Access (SDA) REST API",
        "Auth_Required": "None (Public, but requires Box interface navigation)",
        "License": "US Public Domain",
        "Target_Coverage": "Static/Periodic updates"
    },
    {
        "Source_ID": "SRC-11",
        "Source_Name": "ECMWF / Copernicus Climate Change Service (ERA5-Land)",
        "Organization": "ECMWF / European Union Copernicus Programme",
        "Primary_URL": "https://cds.climate.copernicus.eu/",
        "Data_Type": "9km Hourly & Monthly Reanalysis Climate & Soil Moisture Variables",
        "Access_Method": "CDS API (`cdsapi` Python client)",
        "Auth_Required": "Free Copernicus CDS Account & Personal Access Token (Blocker documented)",
        "License": "Copernicus Open Access Licence",
        "Target_Coverage": "1950 to present (covers 2002-2026)"
    },
    {
        "Source_ID": "SRC-12",
        "Source_Name": "NASA Earthdata / LP DAAC (MODIS Terra/Aqua)",
        "Organization": "NASA Earth Science Data and Information System (ESDIS) / LP DAAC",
        "Primary_URL": "https://search.earthdata.nasa.gov/ & https://appeears.earthdatacloud.nasa.gov/",
        "Data_Type": "MOD13Q1 (250m NDVI/EVI) & MOD15A2H (500m LAI/FPAR)",
        "Access_Method": "AppEEARS REST API & NASA Earthdata Cloud",
        "Auth_Required": "Free NASA Earthdata Login Account & AppEEARS Authorization (Blocker documented)",
        "License": "NASA Open Data Policy (Free and open worldwide)",
        "Target_Coverage": "2000-present (MOD13Q1) & 2002-present (MOD15A2H)"
    },
    {
        "Source_ID": "SRC-13",
        "Source_Name": "USGS EROS / EarthExplorer (Landsat Collection 2)",
        "Organization": "United States Geological Survey (USGS)",
        "Primary_URL": "https://earthexplorer.usgs.gov/ & https://m2m.cr.usgs.gov/",
        "Data_Type": "30m Multispectral Surface Reflectance (Landsat 5/7/8/9)",
        "Access_Method": "USGS M2M API & AWS Open Data S3",
        "Auth_Required": "Free USGS EROS Credentials & M2M Token (Blocker documented)",
        "License": "US Public Domain",
        "Target_Coverage": "2002-2026 (L5: 2002-2011; L7: 2002-present [SLC-off post-2003]; L8: 2013-present; L9: 2021-present)"
    },
    {
        "Source_ID": "SRC-14",
        "Source_Name": "Copernicus Data Space Ecosystem (Sentinel-2 MSI)",
        "Organization": "European Space Agency (ESA) / European Commission",
        "Primary_URL": "https://dataspace.copernicus.eu/",
        "Data_Type": "10-20m Multispectral Surface Reflectance (L2A)",
        "Access_Method": "CDSE OData & STAC APIs",
        "Auth_Required": "Free CDSE Account & OAuth2 Bearer Token (Blocker documented)",
        "License": "Copernicus Open Access",
        "Target_Coverage": "Mid-2015 to present (Cannot cover 2002-mid-2015 period)"
    }
]

with open(source_registry_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        "Source_ID", "Source_Name", "Organization", "Primary_URL", "Data_Type",
        "Access_Method", "Auth_Required", "License", "Target_Coverage"
    ])
    writer.writeheader()
    writer.writerows(sources)

print("Source_Registry.csv written successfully.")

# 2. Data_Dictionary.csv
data_dict_path = os.path.join(admin_dir, "Data_Dictionary.csv")
dict_entries = [
    # Yield
    {"Dataset": "USDA NASS Corn Yield", "Variable_Name": "Year", "Data_Type": "Integer", "Units": "Year (YYYY)", "Description": "Observation crop calendar year (2002-2025)", "Spatial_Resolution": "County", "Temporal_Resolution": "Annual"},
    {"Dataset": "USDA NASS Corn Yield", "Variable_Name": "State", "Data_Type": "String", "Units": "Text Name", "Description": "State name within US Corn Belt (e.g. IOWA, ILLINOIS)", "Spatial_Resolution": "State", "Temporal_Resolution": "Static"},
    {"Dataset": "USDA NASS Corn Yield", "Variable_Name": "State ANSI", "Data_Type": "String (2-digit)", "Units": "FIPS Code", "Description": "Official 2-digit Federal Information Processing Standard state code", "Spatial_Resolution": "State", "Temporal_Resolution": "Static"},
    {"Dataset": "USDA NASS Corn Yield", "Variable_Name": "County", "Data_Type": "String", "Units": "Text Name", "Description": "County name or 'OTHER COUNTIES' residual", "Spatial_Resolution": "County", "Temporal_Resolution": "Static"},
    {"Dataset": "USDA NASS Corn Yield", "Variable_Name": "County ANSI", "Data_Type": "String (3-digit)", "Units": "FIPS Code", "Description": "Official 3-digit Federal Information Processing Standard county code", "Spatial_Resolution": "County", "Temporal_Resolution": "Static"},
    {"Dataset": "USDA NASS Corn Yield", "Variable_Name": "Commodity", "Data_Type": "String", "Units": "Text", "Description": "Crop commodity name ('CORN')", "Spatial_Resolution": "County", "Temporal_Resolution": "Annual"},
    {"Dataset": "USDA NASS Corn Yield", "Variable_Name": "Data Item", "Data_Type": "String", "Units": "Text Specification", "Description": "Precise statistic: 'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE'", "Spatial_Resolution": "County", "Temporal_Resolution": "Annual"},
    {"Dataset": "USDA NASS Corn Yield", "Variable_Name": "Value", "Data_Type": "Float", "Units": "Bushels per Acre (bu/acre)", "Description": "Actual observed corn grain yield (original unaltered source values)", "Spatial_Resolution": "County", "Temporal_Resolution": "Annual"},
    {"Dataset": "USDA NASS Corn Yield", "Variable_Name": "CV (%)", "Data_Type": "Float / Empty", "Units": "Percentage (%)", "Description": "Coefficient of variation for survey statistical reliability", "Spatial_Resolution": "County", "Temporal_Resolution": "Annual"},
    
    # NOAA ClimDiv
    {"Dataset": "NOAA NCEI ClimDiv", "Variable_Name": "State_FIPS", "Data_Type": "String (2-digit)", "Units": "FIPS Code", "Description": "State FIPS identifier", "Spatial_Resolution": "County", "Temporal_Resolution": "Static"},
    {"Dataset": "NOAA NCEI ClimDiv", "Variable_Name": "County_FIPS", "Data_Type": "String (3-digit)", "Units": "FIPS Code", "Description": "County FIPS identifier", "Spatial_Resolution": "County", "Temporal_Resolution": "Static"},
    {"Dataset": "NOAA NCEI ClimDiv", "Variable_Name": "Element_Code", "Data_Type": "String (2-digit)", "Units": "Code", "Description": "Climate element: 01=Precipitation, 02=Mean Temp, 27=Max Temp, 28=Min Temp", "Spatial_Resolution": "County", "Temporal_Resolution": "Static"},
    {"Dataset": "NOAA NCEI ClimDiv", "Variable_Name": "Month_01 to Month_12", "Data_Type": "Float", "Units": "Inches (pcpn) / Deg F (temp)", "Description": "Monthly county observed climate value; missing coded as -9.99 or -99.99", "Spatial_Resolution": "County", "Temporal_Resolution": "Monthly"},
    
    # PRISM
    {"Dataset": "PRISM 4km Grids", "Variable_Name": "ppt", "Data_Type": "Raster Float32", "Units": "Millimeters (mm)", "Description": "Gridded total precipitation at 4km spatial resolution", "Spatial_Resolution": "4 km grid cell", "Temporal_Resolution": "Annual & Monthly"},
    {"Dataset": "PRISM 4km Grids", "Variable_Name": "tmean", "Data_Type": "Raster Float32", "Units": "Degrees Celsius (°C)", "Description": "Gridded mean air temperature", "Spatial_Resolution": "4 km grid cell", "Temporal_Resolution": "Annual & Monthly"},
    {"Dataset": "PRISM 4km Grids", "Variable_Name": "tmin", "Data_Type": "Raster Float32", "Units": "Degrees Celsius (°C)", "Description": "Gridded minimum air temperature", "Spatial_Resolution": "4 km grid cell", "Temporal_Resolution": "Annual & Monthly"},
    {"Dataset": "PRISM 4km Grids", "Variable_Name": "tmax", "Data_Type": "Raster Float32", "Units": "Degrees Celsius (°C)", "Description": "Gridded maximum air temperature", "Spatial_Resolution": "4 km grid cell", "Temporal_Resolution": "Annual & Monthly"},
    
    # RMA Crop Insurance
    {"Dataset": "USDA RMA Summary of Business", "Variable_Name": "Net_Planted_Acres", "Data_Type": "Float", "Units": "Acres", "Description": "Net insured corn planted acreage", "Spatial_Resolution": "County", "Temporal_Resolution": "Annual"},
    {"Dataset": "USDA RMA Summary of Business", "Variable_Name": "Liability", "Data_Type": "Float", "Units": "USD ($)", "Description": "Total crop insurance liability underwritten for corn", "Spatial_Resolution": "County", "Temporal_Resolution": "Annual"},
    {"Dataset": "USDA RMA Summary of Business", "Variable_Name": "Loss_Ratio", "Data_Type": "Float", "Units": "Ratio (Indemnity / Total Premium)", "Description": "Actuarial loss ratio reflecting catastrophic drought, flood, or yield loss", "Spatial_Resolution": "County", "Temporal_Resolution": "Annual"},
    
    # CDL
    {"Dataset": "USDA Cropland Data Layer", "Variable_Name": "CDL_Code", "Data_Type": "Integer (1-255)", "Units": "Categorical Class ID", "Description": "Land cover classification: Code 1 represents Corn for grain", "Spatial_Resolution": "30 meters", "Temporal_Resolution": "Annual"},
    
    # SoilGrids
    {"Dataset": "ISRIC SoilGrids", "Variable_Name": "clay", "Data_Type": "Raster Int16", "Units": "g/kg (permille, 0-1000)", "Description": "Proportion of clay particles (<0.002 mm) in the fine earth fraction", "Spatial_Resolution": "250 meters", "Temporal_Resolution": "Static baseline"},
    {"Dataset": "ISRIC SoilGrids", "Variable_Name": "sand", "Data_Type": "Raster Int16", "Units": "g/kg (permille, 0-1000)", "Description": "Proportion of sand particles (0.05-2 mm) in the fine earth fraction", "Spatial_Resolution": "250 meters", "Temporal_Resolution": "Static baseline"},
    {"Dataset": "ISRIC SoilGrids", "Variable_Name": "silt", "Data_Type": "Raster Int16", "Units": "g/kg (permille, 0-1000)", "Description": "Proportion of silt particles (0.002-0.05 mm) in the fine earth fraction", "Spatial_Resolution": "250 meters", "Temporal_Resolution": "Static baseline"},
    {"Dataset": "ISRIC SoilGrids", "Variable_Name": "soc", "Data_Type": "Raster Int16", "Units": "dg/kg (0.1 g/kg)", "Description": "Soil organic carbon content in fine earth fraction", "Spatial_Resolution": "250 meters", "Temporal_Resolution": "Static baseline"},
    {"Dataset": "ISRIC SoilGrids", "Variable_Name": "phh2o", "Data_Type": "Raster Int16", "Units": "pH x 10", "Description": "Soil pH measured in water", "Spatial_Resolution": "250 meters", "Temporal_Resolution": "Static baseline"},
    {"Dataset": "ISRIC SoilGrids", "Variable_Name": "bdod", "Data_Type": "Raster Int16", "Units": "cg/cm^3 (0.01 g/cm^3)", "Description": "Bulk density of fine earth fraction", "Spatial_Resolution": "250 meters", "Temporal_Resolution": "Static baseline"},
    
    # Crop Progress
    {"Dataset": "USDA NASS Crop Progress", "Variable_Name": "CORN - PROGRESS (PLANTED)", "Data_Type": "Float", "Units": "Percent (%)", "Description": "Cumulative percent of corn crop planted by week ending date", "Spatial_Resolution": "State", "Temporal_Resolution": "Weekly"},
    {"Dataset": "USDA NASS Crop Progress", "Variable_Name": "CORN - PROGRESS (SILKING)", "Data_Type": "Float", "Units": "Percent (%)", "Description": "Cumulative percent of corn crop entering critical reproductive silking stage", "Spatial_Resolution": "State", "Temporal_Resolution": "Weekly"},
    {"Dataset": "USDA NASS Crop Progress", "Variable_Name": "CORN - PROGRESS (HARVESTED)", "Data_Type": "Float", "Units": "Percent (%)", "Description": "Cumulative percent of corn grain harvested by week ending date", "Spatial_Resolution": "State", "Temporal_Resolution": "Weekly"},
]

with open(data_dict_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        "Dataset", "Variable_Name", "Data_Type", "Units", "Description",
        "Spatial_Resolution", "Temporal_Resolution"
    ])
    writer.writeheader()
    writer.writerows(dict_entries)

print("Data_Dictionary.csv written successfully.")

# 3. Data_Coverage_Matrix.csv (2002 to 2026 across all sources)
coverage_matrix_path = os.path.join(admin_dir, "Data_Coverage_Matrix.csv")
years_range = list(range(2002, 2027))

matrix_rows = []
dataset_defs = [
    {
        "Dataset_Name": "USDA NASS Corn Grain Yield",
        "Category": "Yield Target (Primary)",
        "Provider": "USDA NASS",
        "Resolution": "County / Annual",
        "Available_Years": "2002-2025 (24 years)",
        "Missing_Years": "2026 (Unpublished)",
        "Status_2026": "Unpublished (Final county estimates scheduled for release Q1 2027)",
        "Coverage_Rule": lambda y: "Final" if y <= 2025 else "Unpublished"
    },
    {
        "Dataset_Name": "NOAA NCEI County ClimDiv (pcpn, tmpc, tmax, tmin)",
        "Category": "Weather & Climate",
        "Provider": "NOAA NCEI",
        "Resolution": "County / Monthly",
        "Available_Years": "2002-2026 (25 years)",
        "Missing_Years": "None (Months 1-10 complete for 2026)",
        "Status_2026": "Provisional (Months Jan-Oct 2026 observed; Nov-Dec pending calendar completion)",
        "Coverage_Rule": lambda y: "Final" if y <= 2025 else "Provisional (Jan-Oct 2026)"
    },
    {
        "Dataset_Name": "PRISM Gridded Climate (ppt, tmean, tmin, tmax)",
        "Category": "Weather & Climate",
        "Provider": "PRISM Climate Group / OSU",
        "Resolution": "4 km Grid / Annual & Monthly",
        "Available_Years": "2002-2026 (25 years)",
        "Missing_Years": "Annual 2026 (Calendar year incomplete)",
        "Status_2026": "Provisional Monthly (Growing season months 202605-202610 downloaded)",
        "Coverage_Rule": lambda y: "Final Annual" if y <= 2025 else "Provisional Monthly (May-Oct 2026)"
    },
    {
        "Dataset_Name": "NASA POWER Agroclimatology (Solar Radiation, Temp, RH)",
        "Category": "Weather & Climate",
        "Provider": "NASA POWER",
        "Resolution": "Daily Point / Regional Grid",
        "Available_Years": "2002-2026 (25 years)",
        "Missing_Years": "None (Daily through Oct 31, 2026)",
        "Status_2026": "Provisional Observed (Through October 31, 2026)",
        "Coverage_Rule": lambda y: "Final" if y <= 2025 else "Provisional Observed (Through Oct 2026)"
    },
    {
        "Dataset_Name": "USDA RMA Crop Insurance & Planted Acreage",
        "Category": "Agricultural Management",
        "Provider": "USDA RMA",
        "Resolution": "County / Annual",
        "Available_Years": "2002-2026 (25 years)",
        "Missing_Years": "None",
        "Status_2026": "Provisional/Current-Year Business (Subject to end-of-year indemnity reconciliations)",
        "Coverage_Rule": lambda y: "Final" if y <= 2025 else "Provisional Active Business"
    },
    {
        "Dataset_Name": "USDA NASS Corn Crop Progress (Planted, Silking, Harvest)",
        "Category": "Crop Phenology & Management",
        "Provider": "USDA NASS",
        "Resolution": "State / Weekly",
        "Available_Years": "2002-2026 (25 years)",
        "Missing_Years": "None (Full 2026 growing season captured)",
        "Status_2026": "Final Season Observations",
        "Coverage_Rule": lambda y: "Final"
    },
    {
        "Dataset_Name": "USDA NASS Corn Crop Condition",
        "Category": "Crop Phenology & Management",
        "Provider": "USDA NASS",
        "Resolution": "State / Weekly",
        "Available_Years": "2002-2026 (25 years)",
        "Missing_Years": "None (Full 2026 growing season captured)",
        "Status_2026": "Final Season Observations",
        "Coverage_Rule": lambda y: "Final"
    },
    {
        "Dataset_Name": "USDA NASS Fertilizer & Chemical Application",
        "Category": "Nutrient Management",
        "Provider": "USDA NASS",
        "Resolution": "State / Periodic Survey",
        "Available_Years": "2002-2021 (Periodic survey vintages)",
        "Missing_Years": "2022-2026 (Off-survey years)",
        "Status_2026": "Unavailable (NASS Agricultural Chemical Use survey is biennial/triennial for corn)",
        "Coverage_Rule": lambda y: "Surveyed" if y in [2001, 2005, 2010, 2014, 2016, 2018, 2021] else "Off-Survey Year"
    },
    {
        "Dataset_Name": "USDA ERS Fertilizer Use and Price",
        "Category": "Nutrient Management",
        "Provider": "USDA ERS",
        "Resolution": "National / State / Annual",
        "Available_Years": "Historical through 2024",
        "Missing_Years": "2025, 2026",
        "Status_2026": "Unavailable (Awaiting annual compilation update)",
        "Coverage_Rule": lambda y: "Final" if y <= 2024 else "Unavailable"
    },
    {
        "Dataset_Name": "USDA Cropland Data Layer (CDL)",
        "Category": "Crop Classification Mask",
        "Provider": "USDA NASS / CropScape",
        "Resolution": "30m GeoTIFF / Annual",
        "Available_Years": "2008-2025 CONUS (2002-2007 Partial Midwest)",
        "Missing_Years": "2026 (Unpublished)",
        "Status_2026": "Unpublished (Scheduled for publication Q1 2027)",
        "Coverage_Rule": lambda y: "Partial Midwest" if y < 2008 else ("CONUS Complete" if y <= 2025 else "Unpublished")
    },
    {
        "Dataset_Name": "US Census TIGER/Line County Boundaries",
        "Category": "Geographic Reference",
        "Provider": "US Census Bureau",
        "Resolution": "County Vector Polygons",
        "Available_Years": "2024 Definitive Vintage",
        "Missing_Years": "Static Base",
        "Status_2026": "Valid Administrative Reference for 2002-2026 study period",
        "Coverage_Rule": lambda y: "Static Base Valid"
    },
    {
        "Dataset_Name": "ISRIC SoilGrids Physical/Chemical Properties",
        "Category": "Soil Reference",
        "Provider": "ISRIC - World Soil Information",
        "Resolution": "250m Gridded VRT Rasters",
        "Available_Years": "Latest 2.0 Static Baseline",
        "Missing_Years": "Static Base",
        "Status_2026": "Static Pedogenic Reference (Not duplicated artificially)",
        "Coverage_Rule": lambda y: "Static Pedogenic Reference"
    }
]

header = ["Dataset_Name", "Category", "Provider", "Resolution", "Available_Years", "Missing_Years", "Status_2026"] + [str(y) for y in years_range]

with open(coverage_matrix_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for ds in dataset_defs:
        row = [
            ds["Dataset_Name"], ds["Category"], ds["Provider"], ds["Resolution"],
            ds["Available_Years"], ds["Missing_Years"], ds["Status_2026"]
        ]
        for y in years_range:
            row.append(ds["Coverage_Rule"](y))
        writer.writerow(row)

print("Data_Coverage_Matrix.csv written successfully.")

# 4. Missing_Data_Report.csv
missing_rep_path = os.path.join(admin_dir, "Missing_Data_Report.csv")
missing_records = [
    {
        "Dataset": "USDA NASS Corn Grain Yield",
        "Scope_Missing": "Year 2026",
        "Reason": "Crop harvest and county production survey is undergoing post-harvest statistical reconciliation; final county estimates are released in February/March 2027.",
        "Impact": "Primary prediction target labels exist for 2002-2025 (24 continuous years); 2026 serves as an unobserved prospective forecasting benchmark.",
        "Alternative_Remedy": "Provisional state-level USDA WASDE / NASS Crop Production forecasts can provide top-level benchmarking, but county-level labels must never be fabricated."
    },
    {
        "Dataset": "USDA Cropland Data Layer (CDL)",
        "Scope_Missing": "Pre-2008 in select states (2002-2007) & 2026",
        "Reason": "National 30m CDL program achieved full 48-state CONUS coverage in 2008. Prior to 2008, only select major agricultural states (IA, IL, IN, NE, ND) were classified. 2026 raster is unpublished.",
        "Impact": "Pixel-level crop masking for 2002-2007 requires historical state-specific CDL tiles or MODIS/Landsat land cover alternatives; 2026 awaits official NASS release.",
        "Alternative_Remedy": "USGS NLCD (National Land Cover Database) 2001/2004/2006 cultivated cropland masks and MODIS Land Cover Type (MCD12Q1) provide official peer-reviewed land cover."
    },
    {
        "Dataset": "PRISM Annual Weather Grids",
        "Scope_Missing": "Calendar Year 2026 Annual Grid",
        "Reason": "PRISM annual grids (`us_4km_ppt_2026.zip`) are generated only after the complete 365-day calendar year has concluded and undergone quality control.",
        "Impact": "Full annual summary grid unavailable until early 2027.",
        "Alternative_Remedy": "Monthly growing season provisional grids (May through October 2026) have been acquired and verified, providing full weather during the actual maize vegetative and reproductive stages."
    },
    {
        "Dataset": "USDA ERS Fertilizer Use and Price",
        "Scope_Missing": "Years 2025 and 2026",
        "Reason": "ERS updates the comprehensive workbook with a 1-2 year reporting lag based on chemical manufacturing and agricultural retail reporting cycles.",
        "Impact": "Recent macro fertilizer price indices require supplemental indexing from monthly USDA NASS Agricultural Prices reports.",
        "Alternative_Remedy": "NASS Agricultural Prices report indices or World Bank commodity fertilizer price indices."
    },
    {
        "Dataset": "Sentinel-2 MSI",
        "Scope_Missing": "Years 2002 through mid-2015",
        "Reason": "Sentinel-2A was launched in June 2015; data does not physically exist prior to late 2015.",
        "Impact": "Cannot be used as a sole continuous optical feature source across the full 25-year study horizon.",
        "Alternative_Remedy": "MODIS (MOD13Q1 NDVI/EVI, MOD15A2H LAI/FPAR) and Landsat 5/7/8 provide continuous 30m-250m satellite coverage across the entire 2002-2026 timeline."
    },
    {
        "Dataset": "ERA5-Land / MODIS / Landsat / Sentinel-2 Raw Portals",
        "Scope_Missing": "Direct automated download hindered by mandatory user authentication / credentials",
        "Reason": "NASA Earthdata, Copernicus CDS, USGS EROS, and CDSE require individual user registrations, API tokens, and license acceptances.",
        "Impact": "Direct file downloading blocked until user configures API credentials in project environment.",
        "Alternative_Remedy": "Complete technical access specifications, environment file instructions (`~/.netrc`, `~/.cdsapirc`), and Python scripts provided in corresponding raw subdirectories."
    }
]

with open(missing_rep_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["Dataset", "Scope_Missing", "Reason", "Impact", "Alternative_Remedy"])
    writer.writeheader()
    writer.writerows(missing_records)

print("Missing_Data_Report.csv written successfully.")

