# CROP SPECIFIC PIXEL MASKING VALIDATION REPORT

## The Necessity of Cropland Filtering
In county-level remote sensing aggregation, unmasked NDVI/EVI represents the entire spatial landscape, including deciduous forests, urban developments, pastures, and water bodies.
1. Forests in the eastern Corn Belt (Indiana, Ohio) exhibit high and stable NDVI (~0.85) even during agricultural droughts.
2. Unmasked county NDVI severely underestimates drought severity in corn fields because perennial forest canopies mask crop senescence.
3. Therefore, satellite indices must be filtered strictly using the USDA Cropland Data Layer (CDL) Corn Mask (Code 1) to represent actual Zea mays physiological response.
