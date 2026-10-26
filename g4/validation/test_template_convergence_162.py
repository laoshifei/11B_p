"""Controls for generated-ID allocation and bounded three-alpha reduction."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.bayes import template_convergence_162_study as study


class ConvergenceTests(unittest.TestCase):
    def test_nonreacting_incident_is_not_added_to_rejected_reaction_budget(self):
        budgets = study.allocated_budgets(999999, 1)
        self.assertEqual(budgets['fresh_large'], 999999)
        self.assertEqual(budgets['coherent_only'], 999999)
        self.assertEqual(budgets['fresh_small'], 100000)
        with self.assertRaises(ValueError):
            study.allocated_budgets(800000, 1)

    def test_partitions_are_disjoint_before_any_acceptance_and_prefixes_are_nested(self):
        ids = np.arange(101, dtype=np.int64)*3
        template, data = study.partitions(ids, 1)
        self.assertEqual(len(template), 50)
        self.assertEqual(len(data), 51)
        self.assertEqual(len(np.intersect1d(template, data)), 0)
        np.testing.assert_array_equal(np.sort(np.r_[template, data]), ids)
        np.testing.assert_array_equal(template, study.partitions(ids, 1)[0])
        self.assertFalse(np.array_equal(template, study.partitions(ids, 0)[0]))
        self.assertTrue(set(template[:10]) < set(template[:25]) < set(template))
        with self.assertRaises(ValueError):
            study.partitions([2, 2, 3], 0)

    def test_sphere_rows_can_cross_chunks_and_duplicates_invalidate_triplet(self):
        scan = study.mismatch.scan
        # Event 0 is complete across chunks; event 1 duplicates index 2;
        # event 2 has a foreign-source row, which cannot complete its triplet.
        event = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 0])
        index = np.array([0, 1, 0, 1, 2, 2, 0, 1, 2, 2])
        chunk = dict(event_id=event, generator_particle_index=index,
            pdg=np.full(10, scan.ALPHA_PDG),
            h11b_particle_source=np.full(10, scan.SOURCE_H11B_REACTION_PRODUCT))
        chunk["h11b_particle_source"][8] = 999
        for offset, field in enumerate(("kinetic_energy_MeV", "px_MeV_c", "py_MeV_c", "pz_MeV_c", "x_mm", "y_mm", "z_mm")):
            chunk[field] = np.arange(10)+100*offset
        values, counts = np.zeros((3, 3, 7)), np.zeros((3, 3), np.int32)
        for first, last in ((0, 2), (2, 8), (8, 10)):
            study.accumulate_sphere(values, counts, {k:v[first:last] for k,v in chunk.items()})
        np.testing.assert_array_equal(np.all(counts == 1, axis=1), [True, False, False])
        np.testing.assert_array_equal(values[0, :, 0], [0, 1, 9])
        self.assertEqual(counts[1, 2], 2)
        self.assertEqual(counts[2, 2], 0)

    def test_same_budget_regeneration_is_identical_but_actual_bank_can_differ(self):
        truth = [[.7, .2, .1], [.1, .7, .2], [.3, .5, .2]]
        budgets = [1000, 2000, 800]
        a = dict(truth=truth, banks={"coherent":dict(generated=budgets, raw_templates=[[700,200],[200,1400],[240,400]])})
        b = dict(truth=truth, banks={"coherent":dict(generated=budgets, raw_templates=[[680,220],[210,1390],[230,410]])})
        regenerated_a = study.finite.draw(a, .005, study.VALIDATION_SEED, 17, 10000)
        regenerated_b = study.finite.draw(b, .005, study.VALIDATION_SEED, 17, 10000)
        for x, y in zip(regenerated_a, regenerated_b):
            np.testing.assert_array_equal(x, y)
        fixed_a = study.finite.draw(a, .005, study.CONDITIONAL_SEED, 17, 10000, fluctuate_mc=False)
        fixed_b = study.finite.draw(b, .005, study.CONDITIONAL_SEED, 17, 10000, fluctuate_mc=False)
        np.testing.assert_array_equal(fixed_a[0], fixed_b[0])
        self.assertFalse(np.array_equal(fixed_a[1], fixed_b[1]))

    def test_freezing_rejects_validation_seed_and_uses_family_rank(self):
        rows = {m:dict(q_upper=np.linspace(0, 5, 1000), q0=np.linspace(0, 4, 1000))
                for m in study.finite.METHODS}
        state = dict(configuration=dict(seed=study.VALIDATION_SEED,repetitions=1000),
                     records={"00000":rows,"05000":rows,"20000":rows})
        with self.assertRaises(ValueError):
            study.freeze_calibration(state)
        state["configuration"]["seed"] = study.CALIBRATION_SEED
        frozen = study.freeze_calibration(state)
        self.assertEqual(frozen["tolerance"]["comparisons"],16)
        self.assertGreater(frozen["tolerance"]["rank"],950)

    def test_compressed_trials_are_checksum_verified(self):
        records = dict(seed=study.CALIBRATION_SEED,map=np.array([.05,.95,0.]),trials=[0,1,2])
        envelope = study.pack(records)
        self.assertEqual(study.unpack(envelope)["trials"],[0,1,2])
        envelope["payload_sha256"] = "invalid"
        with self.assertRaises(ValueError):
            study.unpack(envelope)


if __name__ == "__main__":
    unittest.main()
