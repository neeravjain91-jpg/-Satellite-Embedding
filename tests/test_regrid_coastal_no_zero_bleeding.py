import os
import sys
import unittest
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.regrid import regrid_2d_field

class TestRegridCoastalNoZeroBleeding(unittest.TestCase):
    """
    Regression test suite for normalized bilinear interpolation:
    Ensures that coastal ocean cells adjacent to land are not artificially pulled
    toward 0.0 by naive NaN-to-zero substitution during regridding.
    """

    def test_coastal_ocean_cell_not_pulled_to_zero(self):
        """
        Verify that an ocean cell adjacent to land maintains its authentic physical
        temperature (e.g. 28.0 °C) rather than being averaged with 0.0 °C land values.
        """
        # Source grid: 4x4 from 10.0 to 13.0 lat, 70.0 to 73.0 lon
        src_lats = np.array([10.0, 11.0, 12.0, 13.0])
        src_lons = np.array([70.0, 71.0, 72.0, 73.0])

        # West side (lon 70, 71) is land (NaN); East side (lon 72, 73) is ocean (28.0 °C)
        data = np.array([
            [np.nan, np.nan, 28.0, 28.0],
            [np.nan, np.nan, 28.0, 28.0],
            [np.nan, np.nan, 28.0, 28.0],
            [np.nan, np.nan, 28.0, 28.0],
        ], dtype=np.float32)

        # Destination query points:
        # 1. Open ocean cell: lat 11.5, lon 72.5 (all 4 surrounding points are ocean)
        # 2. Coastal ocean cell: lat 11.5, lon 71.5 (surrounded by 2 land cells at lon 71 and 2 ocean cells at lon 72)
        dst_lats = np.array([11.5])
        dst_lons = np.array([71.5, 72.5])

        out = regrid_2d_field(data, src_lats, src_lons, dst_lats=dst_lats, dst_lons=dst_lons)

        self.assertEqual(out.dtype, np.float32, "Output array must have dtype float32")
        self.assertEqual(out.shape, (1, 2))

        # Coastal cell at (11.5, 71.5): bilinear weight is 50% ocean (28.0), 50% land.
        # Under normalized interpolation, denominator is 0.5 >= 0.5 (valid ocean mask),
        # and numerator / denominator = (0.5 * 28.0) / 0.5 = 28.0!
        coastal_val = out[0, 0]
        self.assertFalse(np.isnan(coastal_val), "Coastal ocean cell must be valid")
        self.assertAlmostEqual(float(coastal_val), 28.0, delta=0.01,
                               msg=f"Coastal cell was artificially cooled to {coastal_val} °C instead of ~28.0 °C")

        # Open ocean cell at (11.5, 72.5) must also be 28.0 °C
        open_val = out[0, 1]
        self.assertAlmostEqual(float(open_val), 28.0, delta=0.01)

    def test_land_interior_strictly_nan(self):
        """
        Verify that cells deep in land (denominator < 0.5) strictly remain NaN.
        """
        src_lats = np.array([10.0, 11.0, 12.0, 13.0])
        src_lons = np.array([70.0, 71.0, 72.0, 73.0])

        data = np.array([
            [np.nan, np.nan, 28.0, 28.0],
            [np.nan, np.nan, 28.0, 28.0],
            [np.nan, np.nan, 28.0, 28.0],
            [np.nan, np.nan, 28.0, 28.0],
        ], dtype=np.float32)

        # Query well inside land: lon 70.5
        dst_lats = np.array([11.5])
        dst_lons = np.array([70.5])

        out = regrid_2d_field(data, src_lats, src_lons, dst_lats=dst_lats, dst_lons=dst_lons)
        self.assertTrue(np.isnan(out[0, 0]), "Land interior cell must be strictly NaN")

    def test_vector_wind_current_preservation(self):
        """
        Verify vector-component semantics (e.g. negative zonal wind/current) are preserved
        without being artificially pulled toward 0.0 near coasts.
        """
        src_lats = np.array([15.0, 16.0, 17.0])
        src_lons = np.array([60.0, 61.0, 62.0])

        # Negative wind u-component (trade wind = -12.5 m/s)
        wind_u = np.array([
            [np.nan, -12.5, -12.5],
            [np.nan, -12.5, -12.5],
            [np.nan, -12.5, -12.5],
        ], dtype=np.float32)

        dst_lats = np.array([15.5])
        dst_lons = np.array([60.5, 61.5])

        out = regrid_2d_field(wind_u, src_lats, src_lons, dst_lats=dst_lats, dst_lons=dst_lons)

        coastal_u = out[0, 0]
        self.assertFalse(np.isnan(coastal_u))
        self.assertAlmostEqual(float(coastal_u), -12.5, delta=0.01,
                               msg=f"Vector component {coastal_u} m/s was decelerated toward 0 near coast")

    def test_completely_nan_field_handles_zero_division_safely(self):
        """
        Verify that an entirely NaN field is handled without throwing divide-by-zero errors.
        """
        src_lats = np.array([10.0, 11.0])
        src_lons = np.array([70.0, 71.0])
        all_nan = np.full((2, 2), np.nan, dtype=np.float32)

        dst_lats = np.array([10.5])
        dst_lons = np.array([70.5])

        out = regrid_2d_field(all_nan, src_lats, src_lons, dst_lats=dst_lats, dst_lons=dst_lons)
        self.assertEqual(out.shape, (1, 1))
        self.assertTrue(np.isnan(out[0, 0]))

if __name__ == "__main__":
    unittest.main()
