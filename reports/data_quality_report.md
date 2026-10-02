# Automated Data Quality Control (QC) Report

**Domain**: North Indian Ocean (5°N–30°N, 45°E–105°E)

**Total variables evaluated**: 8

## Detailed Variable Assessment

| dataset           | variable   | units   |   total_points |   nan_fraction |   val_min |   val_max |   val_mean | lat_monotonic   | lon_monotonic   | fill_values_detected   | range_valid   | land_ocean_consistent   | overall_status   |
|:------------------|:-----------|:--------|---------------:|---------------:|----------:|----------:|-----------:|:----------------|:----------------|:-----------------------|:--------------|:------------------------|:-----------------|
| Surface_SST       | sst        | degC    |        8908806 |         0.2366 |    26.21  |    29.7   |     27.908 | True            | True            | False                  | True          | True                    | PASS             |
| Surface_SSS       | sss        | psu     |        8908806 |         0.2366 |    32     |    36.5   |     34.897 | True            | True            | False                  | True          | True                    | PASS             |
| Surface_SSH       | ssh        | m       |        8908806 |         0.2366 |    -0.08  |     0.08  |      0.013 | True            | True            | False                  | True          | True                    | PASS             |
| Surface_CURRENT_U | current_u  | m/s     |        8908806 |         0.2366 |    -0.175 |     0.329 |      0.125 | True            | True            | False                  | True          | True                    | PASS             |
| Surface_CURRENT_V | current_v  | m/s     |        8908806 |         0.2366 |    -0.2   |     0.141 |     -0.1   | True            | True            | False                  | True          | True                    | PASS             |
| Surface_WIND_U    | wind_u     | m/s     |        8908806 |         0.2366 |    -5.612 |    -4.5   |     -4.948 | True            | True            | False                  | True          | True                    | PASS             |
| Surface_WIND_V    | wind_v     | m/s     |        8908806 |         0.2366 |    -5.5   |    -4.5   |     -5.246 | True            | True            | False                  | True          | True                    | PASS             |
| Target_GLORYS     | thetao     | degC    |      133632090 |         0.2977 |     5.581 |    29.7   |     18.706 | True            | True            | False                  | True          | True                    | PASS             |

## Verification Summary

- **Passed Variables**: 8
- **Failed Variables**: 0

All variables satisfy coordinate monotonicity, physical bounds, fill-value cleanliness, and land/ocean masking.
