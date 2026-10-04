"""
tests/test_b6_b7_b8_genuine_context.py
Regression tests proving that B6/B7/B8 receive and utilize genuine
spatial/temporal context when the spatial_temporal_context module is used.

These tests directly address the BLOCKER findings from the OPUS audit:
- B6: must receive spatially varying patches (not replicated pointwise features)
- B7: must receive temporally varying sequences (not replicated current-day features)
- B8: must receive both spatial and temporal variation in cubes

Tests also verify:
- Purge boundary compliance in temporal sequences
- Correct patch extraction from padded grids
- Warning emission on degenerate fallback paths
"""

import os
import sys
import warnings
import numpy as np
import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.spatial_temporal_context import (
    _lat_lon_to_grid_indices,
    _normalize_grid,
    build_spatial_patches,
    build_temporal_sequences,
    build_spatiotemporal_cubes,
    augment_split_with_spatial_patches,
    augment_split_with_temporal_sequences,
    augment_split_with_spatiotemporal_cubes,
)
from preprocessing.canonical_grid import CANONICAL_LATS, CANONICAL_LONS


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def synthetic_gridded_surface():
    """
    Creates a synthetic (T, H, W, C) gridded surface array
    with GENUINE spatial and temporal variation.
    
    T = 30 days, H = len(CANONICAL_LATS), W = len(CANONICAL_LONS), C = 7
    
    Spatial variation: each (h, w) pixel has unique features based on
    lat/lon coordinates (not constant across space).
    
    Temporal variation: each day t adds a t-dependent offset to features
    (not constant across time).
    """
    T = 30
    H = len(CANONICAL_LATS)  # 101
    W = len(CANONICAL_LONS)  # 241
    C = 7
    
    surf = np.zeros((T, H, W, C), dtype=np.float32)
    
    # Create genuine spatial variation
    lat_grid = CANONICAL_LATS[:, np.newaxis]  # (101, 1)
    lon_grid = CANONICAL_LONS[np.newaxis, :]  # (1, 241)
    
    for c in range(C):
        # Each feature channel has unique spatial pattern
        spatial_pattern = np.sin(lat_grid * (c + 1) * 0.1) * np.cos(lon_grid * (c + 1) * 0.05)
        for t in range(T):
            # Each day adds unique temporal modulation
            temporal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * t / 30 + c)
            surf[t, :, :, c] = spatial_pattern * temporal_factor + t * 0.01 * (c + 1)
    
    return surf


@pytest.fixture
def scaler():
    """Training scaler statistics (mean/std per channel)."""
    return {
        "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }


@pytest.fixture
def sample_split_data():
    """
    Creates a tabular split dict with samples at known grid positions.
    Uses 10 samples from days 5-14 at varied spatial locations.
    """
    N = 10
    # Pick positions in the interior of the grid (away from edges)
    lat_values = np.array([10.0, 12.5, 15.0, 17.5, 20.0,
                           10.0, 12.5, 15.0, 17.5, 20.0], dtype=np.float32)
    lon_values = np.array([60.0, 65.0, 70.0, 75.0, 80.0,
                           85.0, 90.0, 70.0, 75.0, 80.0], dtype=np.float32)
    time_indices = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.int32)
    
    return {
        "time_idx": time_indices,
        "lat": lat_values,
        "lon": lon_values,
        "X_norm": np.random.randn(N, 7).astype(np.float32),
        "Y": np.random.randn(N, 15).astype(np.float32),
        "mask": np.ones((N, 15), dtype=bool),
    }


# =========================================================================
# TEST 1: Spatial patches have genuine spatial variation
# =========================================================================

class TestGenuineSpatialContext:
    """Verifies that build_spatial_patches extracts genuine multi-pixel patches."""
    
    def test_01_patches_have_spatial_variation(self, synthetic_gridded_surface, scaler):
        """
        CRITICAL REGRESSION TEST:
        An ocean cell's 3×3 patch must contain DIFFERENT feature values at
        different spatial positions. This directly tests the BLOCKER finding
        that Conv2D was seeing spatially constant input.
        """
        surf_norm = _normalize_grid(
            synthetic_gridded_surface,
            np.array(scaler["mean"]), np.array(scaler["std"])
        )
        
        # Sample from interior of grid
        time_indices = np.array([5, 10, 15])
        lat_indices = np.array([40, 50, 60])   # interior positions
        lon_indices = np.array([100, 120, 140])
        
        patches = build_spatial_patches(
            surf_norm, time_indices, lat_indices, lon_indices,
            patch_size=3
        )
        
        assert patches.shape == (3, 7, 3, 3), f"Expected (3,7,3,3) got {patches.shape}"
        
        # KEY ASSERTION: Center pixel must differ from corner pixels
        for i in range(3):
            center = patches[i, :, 1, 1]  # center of 3x3
            corner = patches[i, :, 0, 0]  # top-left corner
            
            # At least some channels must differ between center and corner
            spatial_diff = np.abs(center - corner)
            assert np.any(spatial_diff > 1e-6), (
                f"BLOCKER REGRESSION: Sample {i} has NO spatial variation in patch. "
                f"Center == corner for all channels. Conv2D would see constant input. "
                f"max diff = {np.max(spatial_diff):.2e}"
            )
    
    def test_02_patches_differ_across_samples(self, synthetic_gridded_surface, scaler):
        """Patches at different spatial locations must contain different values."""
        surf_norm = _normalize_grid(
            synthetic_gridded_surface,
            np.array(scaler["mean"]), np.array(scaler["std"])
        )
        
        time_indices = np.array([5, 5])  # same time, different location
        lat_indices = np.array([30, 70])
        lon_indices = np.array([50, 150])
        
        patches = build_spatial_patches(
            surf_norm, time_indices, lat_indices, lon_indices,
            patch_size=3
        )
        
        assert not np.allclose(patches[0], patches[1]), \
            "Patches at different grid locations should contain different values"
    
    def test_03_patch_boundary_handling(self, synthetic_gridded_surface, scaler):
        """Edge/corner grid cells should use edge-padding, not crash."""
        surf_norm = _normalize_grid(
            synthetic_gridded_surface,
            np.array(scaler["mean"]), np.array(scaler["std"])
        )
        
        # Corner positions
        time_indices = np.array([0, 0, 0, 0])
        lat_indices = np.array([0, 0, 100, 100])
        lon_indices = np.array([0, 240, 0, 240])
        
        patches = build_spatial_patches(
            surf_norm, time_indices, lat_indices, lon_indices,
            patch_size=3
        )
        
        assert patches.shape == (4, 7, 3, 3)
        assert np.all(np.isfinite(patches)), "Edge patches should be finite (edge-padded)"
    
    def test_04_augment_adds_X_patches_key(self, synthetic_gridded_surface,
                                            scaler, sample_split_data):
        """augment_split_with_spatial_patches adds 'X_patches' key to split dict."""
        data = augment_split_with_spatial_patches(
            sample_split_data, synthetic_gridded_surface, scaler,
            patch_size=3
        )
        
        assert "X_patches" in data, "X_patches key not added"
        assert data["X_patches"].shape == (10, 7, 3, 3)
        assert data["X_patches"].dtype == np.float32


# =========================================================================
# TEST 2: Temporal sequences have genuine temporal variation
# =========================================================================

class TestGenuineTemporalContext:
    """Verifies that build_temporal_sequences extracts genuine multi-timestep sequences."""
    
    def test_01_sequences_have_temporal_variation(self, synthetic_gridded_surface, scaler):
        """
        CRITICAL REGRESSION TEST:
        A temporal window [t-4, t-3, t-2, t-1, t] must contain DIFFERENT
        feature values at different time steps. This directly tests the
        BLOCKER finding that GRU was seeing temporally constant input.
        """
        surf_norm = _normalize_grid(
            synthetic_gridded_surface,
            np.array(scaler["mean"]), np.array(scaler["std"])
        )
        
        # Sample from day 10 (enough history for window_size=5)
        time_indices = np.array([10, 15, 20])
        lat_indices = np.array([50, 50, 50])
        lon_indices = np.array([120, 120, 120])
        
        sequences = build_temporal_sequences(
            surf_norm, time_indices, lat_indices, lon_indices,
            window_size=5, purge_intervals=()  # no purge in this 30-day test
        )
        
        assert sequences.shape == (3, 5, 7), f"Expected (3,5,7) got {sequences.shape}"
        
        # KEY ASSERTION: Different time steps within window must differ
        for i in range(3):
            step_first = sequences[i, 0, :]  # earliest in window
            step_last = sequences[i, -1, :]  # latest (current day)
            
            temporal_diff = np.abs(step_first - step_last)
            assert np.any(temporal_diff > 1e-6), (
                f"BLOCKER REGRESSION: Sample {i} has NO temporal variation in sequence. "
                f"First timestep == last timestep for all channels. GRU would see constant input. "
                f"max diff = {np.max(temporal_diff):.2e}"
            )
    
    def test_02_sequences_are_causal(self, synthetic_gridded_surface, scaler):
        """
        Window for sample at day t must contain days <= t only.
        No future information.
        """
        surf_norm = _normalize_grid(
            synthetic_gridded_surface,
            np.array(scaler["mean"]), np.array(scaler["std"])
        )
        
        # Sample at day 10, window_size=5 → should contain days [6, 7, 8, 9, 10]
        time_indices = np.array([10])
        lat_indices = np.array([50])
        lon_indices = np.array([120])
        
        sequences = build_temporal_sequences(
            surf_norm, time_indices, lat_indices, lon_indices,
            window_size=5, purge_intervals=()
        )
        
        # Verify by comparing with known grid values
        for w in range(5):
            expected_day = 10 - (5 - 1 - w)  # days 6, 7, 8, 9, 10
            expected_features = surf_norm[expected_day, 50, 120, :]
            np.testing.assert_array_almost_equal(
                sequences[0, w, :], expected_features,
                decimal=5,
                err_msg=f"Window position {w} should contain day {expected_day}"
            )
    
    def test_03_sequences_respect_purge_boundaries(self, synthetic_gridded_surface, scaler):
        """
        Temporal window must NOT cross purge boundaries.
        A validation sample (day >= 259) must not look back into train (day <= 252).
        """
        # Need enough days — extend the fixture conceptually
        # For this test we use a larger T
        T = 366
        H, W, C = 5, 5, 7  # minimal grid
        surf = np.random.randn(T, H, W, C).astype(np.float32)
        
        # Make day-252 features distinct from day-259 features
        surf[252, :, :, :] = -999.0  # training end
        surf[253:259, :, :, :] = -888.0  # purge zone
        
        surf_norm = _normalize_grid(surf, np.zeros(C), np.ones(C))
        
        # Validation sample at day 259 (first val day), window_size=5
        time_indices = np.array([259])
        lat_indices = np.array([2])
        lon_indices = np.array([2])
        
        sequences = build_temporal_sequences(
            surf_norm, time_indices, lat_indices, lon_indices,
            window_size=5,
            purge_intervals=((253, 258), (307, 312))
        )
        
        # No position in the window should contain purge-zone values (-888)
        assert not np.any(np.isclose(sequences[0], -888.0)), \
            "Sequence contains purge-day values — purge boundary violated"
        
        # No position should contain training-end values (-999)
        assert not np.any(np.isclose(sequences[0], -999.0)), \
            "Sequence crossed purge boundary into training partition"
        
        # All positions should be day-259 values (clamped to partition start)
        expected = surf_norm[259, 2, 2, :]
        for w in range(5):
            np.testing.assert_array_almost_equal(
                sequences[0, w, :], expected, decimal=5,
                err_msg=f"Window position {w} should be clamped to day 259 (partition start)"
            )
    
    def test_04_augment_adds_X_seq_key(self, synthetic_gridded_surface,
                                        scaler, sample_split_data):
        """augment_split_with_temporal_sequences adds 'X_seq' key to split dict."""
        data = augment_split_with_temporal_sequences(
            sample_split_data, synthetic_gridded_surface, scaler,
            window_size=5
        )
        
        assert "X_seq" in data, "X_seq key not added"
        assert data["X_seq"].shape == (10, 5, 7)
        assert data["X_seq"].dtype == np.float32


# =========================================================================
# TEST 3: Spatiotemporal cubes have both spatial AND temporal variation
# =========================================================================

class TestGenuineSpatiotemporalContext:
    """Verifies that build_spatiotemporal_cubes produces genuine context."""
    
    def test_01_cubes_have_spatial_and_temporal_variation(self, synthetic_gridded_surface, scaler):
        """
        CRITICAL REGRESSION TEST:
        A spatiotemporal cube [T, C, P, P] must have BOTH:
        - spatial variation (different pixels differ at same time)
        - temporal variation (same pixel differs across time steps)
        """
        surf_norm = _normalize_grid(
            synthetic_gridded_surface,
            np.array(scaler["mean"]), np.array(scaler["std"])
        )
        
        time_indices = np.array([15])
        lat_indices = np.array([50])
        lon_indices = np.array([120])
        
        cubes = build_spatiotemporal_cubes(
            surf_norm, time_indices, lat_indices, lon_indices,
            patch_size=3, window_size=5, purge_intervals=()
        )
        
        assert cubes.shape == (1, 5, 7, 3, 3)
        
        # Spatial variation check: center vs corner at same time step
        cube = cubes[0]  # (5, 7, 3, 3)
        center = cube[-1, :, 1, 1]  # latest time, center pixel
        corner = cube[-1, :, 0, 0]  # latest time, corner pixel
        spatial_diff = np.abs(center - corner)
        assert np.any(spatial_diff > 1e-6), \
            "BLOCKER REGRESSION: No spatial variation in cube"
        
        # Temporal variation check: same pixel at first vs last time step
        first_t = cube[0, :, 1, 1]
        last_t = cube[-1, :, 1, 1]
        temporal_diff = np.abs(first_t - last_t)
        assert np.any(temporal_diff > 1e-6), \
            "BLOCKER REGRESSION: No temporal variation in cube"
    
    def test_02_augment_adds_X_cubes_key(self, synthetic_gridded_surface,
                                          scaler, sample_split_data):
        """augment_split_with_spatiotemporal_cubes adds 'X_cubes' key."""
        data = augment_split_with_spatiotemporal_cubes(
            sample_split_data, synthetic_gridded_surface, scaler,
            patch_size=3, window_size=5
        )
        
        assert "X_cubes" in data, "X_cubes key not added"
        assert data["X_cubes"].shape == (10, 5, 7, 3, 3)
        assert data["X_cubes"].dtype == np.float32


# =========================================================================
# TEST 4: _prepare_* methods prefer genuine context keys over replication
# =========================================================================

class TestPrepareMethodsPreferGenuineContext:
    """Verifies that B6/B7/B8 _prepare_* methods use X_patches/X_seq/X_cubes
    when available, and warn on degenerate fallback."""
    
    def test_01_b6_uses_X_patches_key(self):
        """B6._prepare_patches should use X_patches when available."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        from models.baselines import B6_SpatialCNN
        model = B6_SpatialCNN(patch_size=3)
        
        N, C, P = 5, 7, 3
        genuine_patches = np.random.randn(N, C, P, P).astype(np.float32)
        data = {
            "X_patches": genuine_patches,
            "X_norm": np.random.randn(N, C).astype(np.float32),
        }
        
        # Should use X_patches, NOT X_norm
        result = model._prepare_patches(data)
        np.testing.assert_array_almost_equal(
            result.numpy(), genuine_patches,
            err_msg="B6 should prefer X_patches over X_norm"
        )
    
    def test_02_b7_uses_X_seq_key(self):
        """B7._prepare_sequences should use X_seq when available."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        from models.baselines import B7_TemporalModel
        model = B7_TemporalModel(window_size=5)
        
        N, T, C = 5, 5, 7
        genuine_seq = np.random.randn(N, T, C).astype(np.float32)
        data = {
            "X_seq": genuine_seq,
            "X_norm": np.random.randn(N, C).astype(np.float32),
        }
        
        result = model._prepare_sequences(data)
        np.testing.assert_array_almost_equal(
            result.numpy(), genuine_seq,
            err_msg="B7 should prefer X_seq over X_norm"
        )
    
    def test_03_b8_uses_X_cubes_key(self):
        """B8._prepare_cubes should use X_cubes when available."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        from models.baselines import B8_EmbeddingModel
        model = B8_EmbeddingModel(patch_size=3, window_size=5)
        
        N, T, C, P = 5, 5, 7, 3
        genuine_cubes = np.random.randn(N, T, C, P, P).astype(np.float32)
        data = {
            "X_cubes": genuine_cubes,
            "X_norm": np.random.randn(N, C).astype(np.float32),
        }
        
        result = model._prepare_cubes(data)
        np.testing.assert_array_almost_equal(
            result.numpy(), genuine_cubes,
            err_msg="B8 should prefer X_cubes over X_norm"
        )
    
    def test_04_b6_warns_on_degenerate_fallback(self):
        """B6 must emit UserWarning when using degenerate replication."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        from models.baselines import B6_SpatialCNN
        model = B6_SpatialCNN(patch_size=3)
        
        # Only provide X_norm (no X_patches) — should trigger warning
        data = {"X_norm": np.random.randn(5, 7).astype(np.float32)}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model._prepare_patches(data)
            
            degenerate_warnings = [x for x in w if "DEGENERATE" in str(x.message)]
            assert len(degenerate_warnings) >= 1, \
                "B6 should warn about DEGENERATE spatial context when X_patches is missing"
    
    def test_05_b7_warns_on_degenerate_fallback(self):
        """B7 must emit UserWarning when using degenerate replication."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        from models.baselines import B7_TemporalModel
        model = B7_TemporalModel(window_size=5)
        
        data = {"X_norm": np.random.randn(5, 7).astype(np.float32)}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model._prepare_sequences(data)
            
            degenerate_warnings = [x for x in w if "DEGENERATE" in str(x.message)]
            assert len(degenerate_warnings) >= 1, \
                "B7 should warn about DEGENERATE temporal context when X_seq is missing"
    
    def test_06_b8_warns_on_degenerate_fallback(self):
        """B8 must emit UserWarning when using degenerate replication."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        from models.baselines import B8_EmbeddingModel
        model = B8_EmbeddingModel(patch_size=3, window_size=5)
        
        data = {"X_norm": np.random.randn(5, 7).astype(np.float32)}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model._prepare_cubes(data)
            
            degenerate_warnings = [x for x in w if "DEGENERATE" in str(x.message)]
            assert len(degenerate_warnings) >= 1, \
                "B8 should warn about DEGENERATE spatiotemporal context when X_cubes is missing"


# =========================================================================
# TEST 5: Grid index conversion
# =========================================================================

class TestGridIndexConversion:
    """Verifies coordinate-to-index conversion for canonical grid."""
    
    def test_01_known_positions(self):
        """Known lat/lon values produce correct grid indices."""
        lats = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        lons = np.array([45.0, 60.0, 75.0, 90.0, 105.0, 45.0])
        
        lat_idx, lon_idx = _lat_lon_to_grid_indices(lats, lons)
        
        # lat 5.0 → index 0, lat 10.0 → index 20, etc.
        np.testing.assert_array_equal(lat_idx, [0, 20, 40, 60, 80, 100])
        # lon 45.0 → index 0, lon 60.0 → index 60, etc.
        np.testing.assert_array_equal(lon_idx, [0, 60, 120, 180, 240, 0])
    
    def test_02_quarter_degree_precision(self):
        """0.25° grid positions produce integer indices."""
        lats = np.array([5.25, 10.50, 15.75])
        lons = np.array([45.25, 60.50, 75.75])
        
        lat_idx, lon_idx = _lat_lon_to_grid_indices(lats, lons)
        
        np.testing.assert_array_equal(lat_idx, [1, 22, 43])
        np.testing.assert_array_equal(lon_idx, [1, 62, 123])


# =========================================================================
# TEST 6: End-to-end B6/B7/B8 training with genuine context
# =========================================================================

class TestEndToEndGenuineTraining:
    """Verifies that B6/B7/B8 can train on genuine context without errors."""
    
    def test_01_b6_trains_with_genuine_patches(self, synthetic_gridded_surface, scaler):
        """B6 trains without error when given genuine spatial patches."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        from models.baselines import B6_SpatialCNN
        
        surf_norm = _normalize_grid(
            synthetic_gridded_surface,
            np.array(scaler["mean"]), np.array(scaler["std"])
        )
        
        N = 50
        time_indices = np.random.randint(0, 30, size=N)
        lat_indices = np.random.randint(5, 96, size=N)  # interior
        lon_indices = np.random.randint(5, 236, size=N)
        
        patches = build_spatial_patches(surf_norm, time_indices, lat_indices, lon_indices, patch_size=3)
        
        train_data = {
            "X_patches": patches,
            "Y": np.random.randn(N, 15).astype(np.float32),
            "mask": np.ones((N, 15), dtype=bool),
        }
        
        model = B6_SpatialCNN(patch_size=3)
        # Should not warn about degenerate context
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(train_data, epochs=1, batch_size=16)
            degenerate_warnings = [x for x in w if "DEGENERATE" in str(x.message)]
            assert len(degenerate_warnings) == 0, \
                "B6 should not warn when genuine X_patches are provided"
    
    def test_02_b7_trains_with_genuine_sequences(self, synthetic_gridded_surface, scaler):
        """B7 trains without error when given genuine temporal sequences."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        from models.baselines import B7_TemporalModel
        
        surf_norm = _normalize_grid(
            synthetic_gridded_surface,
            np.array(scaler["mean"]), np.array(scaler["std"])
        )
        
        N = 50
        time_indices = np.random.randint(5, 30, size=N)
        lat_indices = np.random.randint(5, 96, size=N)
        lon_indices = np.random.randint(5, 236, size=N)
        
        sequences = build_temporal_sequences(
            surf_norm, time_indices, lat_indices, lon_indices,
            window_size=5, purge_intervals=()
        )
        
        train_data = {
            "X_seq": sequences,
            "Y": np.random.randn(N, 15).astype(np.float32),
            "mask": np.ones((N, 15), dtype=bool),
        }
        
        model = B7_TemporalModel(window_size=5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(train_data, epochs=1, batch_size=16)
            degenerate_warnings = [x for x in w if "DEGENERATE" in str(x.message)]
            assert len(degenerate_warnings) == 0, \
                "B7 should not warn when genuine X_seq are provided"
    
    def test_03_b8_trains_with_genuine_cubes(self, synthetic_gridded_surface, scaler):
        """B8 trains without error when given genuine spatiotemporal cubes."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        from models.baselines import B8_EmbeddingModel
        
        surf_norm = _normalize_grid(
            synthetic_gridded_surface,
            np.array(scaler["mean"]), np.array(scaler["std"])
        )
        
        N = 50
        time_indices = np.random.randint(5, 30, size=N)
        lat_indices = np.random.randint(5, 96, size=N)
        lon_indices = np.random.randint(5, 236, size=N)
        
        cubes = build_spatiotemporal_cubes(
            surf_norm, time_indices, lat_indices, lon_indices,
            patch_size=3, window_size=5, purge_intervals=()
        )
        
        train_data = {
            "X_cubes": cubes,
            "Y": np.random.randn(N, 15).astype(np.float32),
            "mask": np.ones((N, 15), dtype=bool),
        }
        
        model = B8_EmbeddingModel(patch_size=3, window_size=5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(train_data, epochs=1, batch_size=16)
            degenerate_warnings = [x for x in w if "DEGENERATE" in str(x.message)]
            assert len(degenerate_warnings) == 0, \
                "B8 should not warn when genuine X_cubes are provided"


# =========================================================================
# TEST 7: Pre-Training Gate Hard-Fail and Context Verification
# =========================================================================

class TestPreTrainingGateContextHardFail:
    """
    Verifies that missing or degenerate B6/B7/B8 context HARD-FAILS
    the pre-training acceptance gate, exactly fulfilling Priorities 5, 6, 7, 8, 9, 10.
    """

    def _build_valid_base_dataset(self, tmp_path, synthetic_gridded_surface, scaler):
        """Helper to create a fully valid dataset conforming to locked protocol."""
        from preprocessing.canonical_grid import CANONICAL_DEPTHS, CANONICAL_FEATURES

        cert_file = os.path.join(tmp_path, "cert.json")
        with open(cert_file, "w") as f:
            import json
            json.dump({"acceptance_certified": True}, f)

        manifest_file = "data/manifests/download_manifest.json"

        N_train, N_val, N_test = 60, 20, 20
        surf_norm = _normalize_grid(
            synthetic_gridded_surface,
            np.array(scaler["mean"]), np.array(scaler["std"])
        )

        def make_split(N, t_min, t_max):
            time_idx = np.random.randint(t_min, t_max, size=N)
            lat_idx = np.random.randint(5, 95, size=N)
            lon_idx = np.random.randint(5, 235, size=N)

            patches = build_spatial_patches(surf_norm, time_idx, lat_idx, lon_idx, patch_size=3)
            seq = build_temporal_sequences(surf_norm, time_idx, lat_idx, lon_idx, window_size=5, purge_intervals=())
            cubes = build_spatiotemporal_cubes(surf_norm, time_idx, lat_idx, lon_idx, patch_size=3, window_size=5, purge_intervals=())

            return {
                "X": np.random.randn(N, 7).astype(np.float32),
                "X_norm": np.random.randn(N, 7).astype(np.float32),
                "Y": (np.random.randn(N, len(CANONICAL_DEPTHS)) * 2.0 + 20.0).astype(np.float32),
                "mask": np.ones((N, len(CANONICAL_DEPTHS)), dtype=bool),
                "lat": CANONICAL_LATS[lat_idx],
                "lon": CANONICAL_LONS[lon_idx],
                "time_idx": time_idx,
                "X_patches": patches,
                "X_seq": seq,
                "X_cubes": cubes,
            }

        dataset = {
            "train": make_split(N_train, 0, 25),
            "val": make_split(N_val, 26, 28),
            "test": make_split(N_test, 28, 30),
            "depth_levels": CANONICAL_DEPTHS,
            "scaler": {
                "mean": scaler["mean"],
                "std": scaler["std"],
                "scaler_sha256": "abcdef1234567890"
            },
            "split_metadata": {
                "total_days": 30,
                "purge_buffer_days": 0,
                "train": {"n_days": 26, "n_samples": N_train},
                "val": {"n_days": 2, "n_samples": N_val},
                "test": {"n_days": 2, "n_samples": N_test}
            }
        }
        return dataset, cert_file

    def test_01_missing_context_hard_fails_gate(self, tmp_path, synthetic_gridded_surface, scaler):
        """Pre-training gate must raise PreTrainingGateBlockedError when context keys are missing."""
        from models.pre_training_gate import verify_pre_training_gate, PreTrainingGateBlockedError

        dataset, cert_file = self._build_valid_base_dataset(tmp_path, synthetic_gridded_surface, scaler)

        # Remove X_patches from train split
        del dataset["train"]["X_patches"]

        with pytest.raises(PreTrainingGateBlockedError) as exc_info:
            verify_pre_training_gate(dataset, certificate_path=cert_file, full_year_mode=False)

        assert "Check 9 FAIL" in str(exc_info.value)
        assert "X_patches" in str(exc_info.value)

    def test_02_degenerate_spatial_context_hard_fails_gate(self, tmp_path, synthetic_gridded_surface, scaler):
        """Pre-training gate must hard-fail if X_patches has zero spatial variation across patch."""
        from models.pre_training_gate import verify_pre_training_gate, PreTrainingGateBlockedError

        dataset, cert_file = self._build_valid_base_dataset(tmp_path, synthetic_gridded_surface, scaler)

        # Replace X_patches with degenerate replicated pointwise features
        N = len(dataset["train"]["X_norm"])
        degen_patches = np.repeat(np.repeat(
            dataset["train"]["X_norm"][:, :, np.newaxis, np.newaxis], 3, axis=2), 3, axis=3)
        dataset["train"]["X_patches"] = degen_patches

        with pytest.raises(PreTrainingGateBlockedError) as exc_info:
            verify_pre_training_gate(dataset, certificate_path=cert_file, full_year_mode=False)

        assert "Check 9 FAIL" in str(exc_info.value)
        assert "spatially constant across patch" in str(exc_info.value)

    def test_03_degenerate_temporal_context_hard_fails_gate(self, tmp_path, synthetic_gridded_surface, scaler):
        """Pre-training gate must hard-fail if X_seq has zero temporal variation across window."""
        from models.pre_training_gate import verify_pre_training_gate, PreTrainingGateBlockedError

        dataset, cert_file = self._build_valid_base_dataset(tmp_path, synthetic_gridded_surface, scaler)

        # Replace X_seq with degenerate replicated features
        degen_seq = np.repeat(dataset["train"]["X_norm"][:, np.newaxis, :], 5, axis=1)
        dataset["train"]["X_seq"] = degen_seq

        with pytest.raises(PreTrainingGateBlockedError) as exc_info:
            verify_pre_training_gate(dataset, certificate_path=cert_file, full_year_mode=False)

        assert "Check 9 FAIL" in str(exc_info.value)
        assert "temporally constant across sequence" in str(exc_info.value)

    def test_04_genuine_context_passes_gate(self, tmp_path, synthetic_gridded_surface, scaler):
        """Pre-training gate passes when genuine non-degenerate context tensors are present."""
        from models.pre_training_gate import verify_pre_training_gate

        dataset, cert_file = self._build_valid_base_dataset(tmp_path, synthetic_gridded_surface, scaler)

        report = verify_pre_training_gate(dataset, certificate_path=cert_file, full_year_mode=False)
        assert report["gate_passed"] is True
        assert "X_patches" in report["context_tensors_verified"]
        assert "X_seq" in report["context_tensors_verified"]
        assert "X_cubes" in report["context_tensors_verified"]

    def test_05_no_target_information_in_context_tensors(self, synthetic_gridded_surface, scaler, sample_split_data):
        """
        PRIORITY 7: Confirms that NO target information (thetao / Y) is ever encoded
        or accessible in any context tensor.
        """
        # Augment split with context
        data = augment_split_with_spatial_patches(dict(sample_split_data), synthetic_gridded_surface, scaler)
        data = augment_split_with_temporal_sequences(data, synthetic_gridded_surface, scaler)
        data = augment_split_with_spatiotemporal_cubes(data, synthetic_gridded_surface, scaler)

        # Check feature dimension of context tensors
        assert data["X_patches"].shape[1] == 7, "X_patches must strictly have 7 surface channels"
        assert data["X_seq"].shape[2] == 7, "X_seq must strictly have 7 surface channels"
        assert data["X_cubes"].shape[2] == 7, "X_cubes must strictly have 7 surface channels"

        # Tampering with target Y must produce ZERO change in any context tensor
        tampered_split = dict(sample_split_data)
        tampered_split["Y"] = sample_split_data["Y"] + 9999.0  # huge offset to targets

        data_t = augment_split_with_spatial_patches(dict(tampered_split), synthetic_gridded_surface, scaler)
        data_t = augment_split_with_temporal_sequences(data_t, synthetic_gridded_surface, scaler)
        data_t = augment_split_with_spatiotemporal_cubes(data_t, synthetic_gridded_surface, scaler)

        np.testing.assert_array_equal(data["X_patches"], data_t["X_patches"],
                                      err_msg="Target tampering altered X_patches!")
        np.testing.assert_array_equal(data["X_seq"], data_t["X_seq"],
                                      err_msg="Target tampering altered X_seq!")
        np.testing.assert_array_equal(data["X_cubes"], data_t["X_cubes"],
                                      err_msg="Target tampering altered X_cubes!")

    def test_06_purge_boundaries_strictly_preserved(self, synthetic_gridded_surface, scaler):
        """
        PRIORITY 8: Confirms that 7-day purge boundaries are strictly enforced.
        Sequences from validation or test NEVER reach across purge intervals.
        """
        T = 366
        surf = np.zeros((T, 10, 10, 7), dtype=np.float32)
        surf[:253] = 1.0     # Train partition
        surf[253:259] = 99.0  # Purge 1 (days 253-258)
        surf[259:307] = 2.0  # Validation partition
        surf[307:313] = 99.0  # Purge 2 (days 307-312)
        surf[313:] = 3.0     # Test partition

        surf_norm = _normalize_grid(surf, np.zeros(7), np.ones(7))

        # First validation day: day 259. Window = 5 days.
        val_t = np.array([259, 260, 261])
        lat_idx = np.array([2, 3, 4])
        lon_idx = np.array([2, 3, 4])

        seq_val = build_temporal_sequences(
            surf_norm, val_t, lat_idx, lon_idx,
            window_size=5, purge_intervals=((253, 258), (307, 312))
        )

        # Neither purge values (99.0) nor training values (1.0) must enter validation window
        assert not np.any(np.isclose(seq_val, 99.0)), "Purge buffer leaked into validation temporal sequence!"
        assert not np.any(np.isclose(seq_val, 1.0)), "Training data leaked into validation temporal sequence!"

        # First test day: day 313. Window = 5 days.
        test_t = np.array([313, 314, 315])
        seq_test = build_temporal_sequences(
            surf_norm, test_t, lat_idx, lon_idx,
            window_size=5, purge_intervals=((253, 258), (307, 312))
        )
        assert not np.any(np.isclose(seq_test, 99.0)), "Purge buffer leaked into test temporal sequence!"
        assert not np.any(np.isclose(seq_test, 2.0)), "Validation data leaked into test temporal sequence!"

