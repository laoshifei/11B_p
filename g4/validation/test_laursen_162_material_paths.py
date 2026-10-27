"""Observable geometry bounds, layer isolation and conditional tail accounting."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.efficiency import laursen_162_material_paths as study


class MaterialPathsTests(unittest.TestCase):
    def inputs(self, size=1):
        matched = study.matching.allocate(size)
        matched['detector'][:] = [0, 1, -1]
        matched['multiplicity'][:] = 2
        matched['pixel'][:] = [[0., 7.], [5.3, 8.], [0., 0.]]
        matched['energy'][:] = [2., 2., 0.]
        return matched

    def test_exact_plane_intersection_identifies_only_d1_first_strip(self):
        inventory = study.geometry_inventory()
        records = np.asarray(inventory['direct_pixels_crossing'])
        np.testing.assert_array_equal(records[:, 0], 0)
        np.testing.assert_array_equal(records[:, 1], 0)
        np.testing.assert_array_equal(records[:, 2], np.arange(16))
        self.assertAlmostEqual(inventory['foil_intersection_xprime_mm'][0], -23.59912324810499, places=10)
        self.assertLess(inventory['foil_intersection_xprime_mm'][1], -25.)

    def test_continuous_pixel_and_beam_samples_stay_inside_cosine_bounds(self):
        n = 3000; matched = self.inputs(n)
        rng = np.random.default_rng(16227)
        matched['pixel'][:, :2] = rng.uniform(0., 15., (n, 2, 2))
        bounds = study.pixel_foil_bounds(matched, .1)
        _, horizontal, vertical = study.matching.audit.basis('equation_4_1')
        d = np.maximum(matched['detector'], 0)
        local = rng.uniform(bounds['low_mm'], bounds['high_mm'])
        impact = study.response.CENTRES[d]+local[..., 0, None]*horizontal[d]+local[..., 1, None]*vertical
        xy = rng.uniform(-1., 1., (n, 2))
        normal = study.response.FOIL_NORMAL
        origins = np.column_stack([xy, -normal[0]/normal[2]*xy[:, 0]])
        direction = study.response.unit(impact-origins[:, None, :])
        cosine = abs(direction@normal)
        self.assertTrue(np.all(cosine >= bounds['cosine_low']-1e-13))
        self.assertTrue(np.all(cosine <= bounds['cosine_high']+1e-13))

    def test_crossing_flag_is_independent_of_in_plane_beam_position_extent(self):
        matched = self.inputs()
        small = study.pixel_foil_bounds(matched, .1, 0.)
        large = study.pixel_foil_bounds(matched, .1, 4.)
        np.testing.assert_array_equal(small['crosses'], large['crosses'])
        self.assertTrue(small['crosses'][0, 0])
        self.assertFalse(small['crosses'][0, 1])
        self.assertEqual(small['cosine_low'][0, 0], 0.)
        self.assertLessEqual(large['cosine_low'][0, 1], small['cosine_low'][0, 1])

    def test_shared_charge_can_arrive_from_beyond_surviving_strip_edge(self):
        matched = self.inputs()
        matched['pixel'][0, 0] = [5., 7.]
        physical = study.pixel_foil_bounds(matched, 0.)
        expanded = study.pixel_foil_bounds(matched, .1)
        true_x = 6*study.response.PITCH-study.response.HALF_SIZE+.03
        self.assertGreater(true_x, physical['high_mm'][0, 0, 0])
        self.assertLess(true_x, expanded['high_mm'][0, 0, 0])
        matched['pixel'][0, 0] = [0., 15.]
        edge = study.pixel_foil_bounds(matched, .1)
        self.assertEqual(edge['low_mm'][0, 0, 0], -25.)
        self.assertEqual(edge['high_mm'][0, 0, 1], 25.)

    def test_invalid_support_cannot_be_silently_clamped(self):
        matched = self.inputs()
        with self.assertRaises(ValueError):study.pixel_foil_bounds(matched, study.response.PITCH)
        with self.assertRaises(ValueError):study.pixel_foil_bounds(matched, beam_width_mm=-1.)
        matched['pixel'][0, 0, 0] = 16.
        with self.assertRaises(ValueError):study.pixel_foil_bounds(matched)

    def test_event_angle_uses_only_measured_hits_and_keeps_all_bins(self):
        normal = study.response.FOIL_NORMAL
        tangent = np.cross(normal, [0., 1., 0.])
        angle = np.radians((study.ANGLE_EDGES[1:]+study.ANGLE_EDGES[:-1])/2)
        first = np.sin(angle)[:, None]*normal+np.cos(angle)[:, None]*tangent
        directions = np.stack([first, np.broadcast_to(normal, first.shape), np.broadcast_to(tangent, first.shape)], axis=1)
        active = np.broadcast_to([True, True, False], directions.shape[:2])
        np.testing.assert_array_equal(study.angle_bucket(directions, active), np.arange(len(angle)))

    def tables(self):
        # An exactly invertible linear stopping law keeps this isolation test
        # independent of the production spline tables and their interpolation.
        factors = {'G4_B': 1300., 'G4_C': 1100., 'G4_Si': 900.}
        return SimpleNamespace(table={'materials': {'G4_Si': {'density_g_cm3': 2.33}}},
            restore=lambda energy, column, material: energy+factors[material]*column)

    def test_depth_change_affects_only_boron_when_angles_are_identical(self):
        matched = self.inputs(); matched['pixel'][0, 0] = [6., 7.]
        tables = self.tables()
        direction, _ = study.decomposition.nominal_measurements(matched, tables)
        true_columns = study.matching.audit.columns(direction, matched['detector'], np.array([.1]),
            study.response.Response(), tables, 'equation_4_1')
        residual, _ = study.material_controls(matched, direction, true_columns, tables)
        np.testing.assert_allclose(residual[:, [2, 4, 6, 8, 9, 10]], np.repeat(residual[:, :1], 6, axis=1), atol=1e-11, rtol=0)
        np.testing.assert_allclose(residual[:, [3, 5, 7]], np.repeat(residual[:, 1:2], 3, axis=1), atol=1e-11, rtol=0)
        self.assertGreater(abs(residual[0, 1]-residual[0, 0]), .001)

    def test_carbon_side_and_angle_controls_are_distinct(self):
        matched = self.inputs(); matched['pixel'][0, 0] = [6., 7.]
        tables = self.tables()
        direction, _ = study.decomposition.nominal_measurements(matched, tables)
        normal = study.response.FOIL_NORMAL
        reflected = direction-2*(direction@normal)[..., None]*normal
        true_columns = study.matching.audit.columns(reflected, matched['detector'], np.array([.5]),
            study.response.Response(), tables, 'equation_4_1')
        residual, _ = study.material_controls(matched, reflected, true_columns, tables)
        np.testing.assert_allclose(residual[:, 9], residual[:, 0], atol=1e-11)
        np.testing.assert_allclose(residual[:, 10], residual[:, 2], atol=1e-11)
        self.assertGreater(abs(residual[0, 2]-residual[0, 0]), 1e-4)

    def test_partition_means_errors_and_tail_mass_match_direct_calculation(self):
        weights = np.array([[1., 4.], [2., 3.], [3., 2.], [4., 1.]])
        residual = np.repeat(np.array([-.1, 1.5, -1.2, .3])[:, None], len(study.CASES), axis=1)
        residual[:, 1] = [1.1, .2, -.1, -.3]
        partitions = np.array([0, 1, 0, 1])
        item = study.PartitionMoments(2, 2)
        item.add(weights, np.zeros(4, int), partitions, residual)
        result = item.finish(weights.sum(axis=0))
        for model in range(2):
            for part in (0, 1):
                w = weights[partitions == part, model]; r = residual[partitions == part]
                mean = w@r/w.sum(); keep = (abs(r) <= .4).astype(float)
                difference = keep-keep[:, :1]; change = w@difference/w.sum()
                error = np.sqrt((w*w)@((difference-change)**2))/w.sum()
                np.testing.assert_allclose(result['mean_MeV'][model, 0, part], mean)
                np.testing.assert_allclose(result['paired_retention_change'][model, 0, part], change)
                np.testing.assert_allclose(result['paired_retention_standard_error'][model, 0, part], error)
            fraction = result['cohort_fraction'][model, 0]
            pooled = fraction@result['rms_MeV'][model, 0]**2
            np.testing.assert_allclose(pooled, weights[:, model]@(residual*residual)/weights[:, model].sum())
            tail = fraction@result['tail1_probability'][model, 0]
            np.testing.assert_allclose(tail, weights[:, model]@(abs(residual) > 1)/weights[:, model].sum())
        self.assertFalse(result['valid'][:, 1:].any())

    def test_nonfinite_residual_requires_explicit_common_cohort(self):
        item = study.PartitionMoments(1, 1)
        residual = np.zeros((1, len(study.CASES))); residual[0, 4] = np.nan
        with self.assertRaises(ValueError):
            item.add(np.ones((1, 1)), np.array([0]), np.array([0]), residual)


if __name__ == '__main__':
    unittest.main()
