"""Independent geometry, loss, probability and reconstruction checks."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.efficiency import laursen_162_response as r


def constant_tables():
    rows=[[.001,800.,200.],[.1,800.,200.],[1.,800.,200.],[10.,800.,200.]]
    return r.RangeTables(dict(materials={m:dict(rows=rows,density_g_cm3=2.)
                                         for m in ('G4_B','G4_C','G4_Si')}))


class LiteratureResponseTests(unittest.TestCase):
    def test_point_rays_hit_known_centres_and_miss_along_vertical(self):
        directions=np.array([[r.unit(r.CENTRES[0]),r.unit(r.CENTRES[1]),[0,1,0]]])
        traced=r.trace(directions,np.zeros((1,3)))
        np.testing.assert_array_equal(traced['detector'],[[0,1,-1]])
        self.assertTrue(np.all(traced['pixel'][0,:2]>=0))
        self.assertTrue(np.all(traced['pixel'][0,:2]<16))

    def test_parallel_and_away_rays_are_not_intersections_behind_origin(self):
        origins=np.array([[100.,0.,0.]])
        rays=np.array([[[1.,0.,0.],[0.,1.,0.],r.HORIZONTAL[0]]])
        got=r.trace(rays,origins)
        self.assertEqual(got['detector'][0,0],-1)
        self.assertEqual(got['detector'][0,1],-1)

    def test_rectangular_solid_angle_matches_known_on_axis_formula(self):
        z=30.
        expected=4*np.arctan(25**2/(z*np.sqrt(z*z+2*25**2)))
        self.assertAlmostEqual(r.rectangle_solid_angle([0,0,z],[0,0,1],[1,0,0]),expected,places=13)

    def test_isotropic_rays_match_independent_rectangle_integral(self):
        rng=np.random.default_rng(260162)
        rays=r.unit(rng.normal(size=(100000,3,3)))
        hits=r.trace(rays,np.zeros((len(rays),3)))['detector']
        for d in (0,1):
            expected=r.rectangle_solid_angle(r.CENTRES[d],r.NORMALS[d],r.HORIZONTAL[d])/(4*np.pi)
            measured=np.mean(hits==d)
            self.assertLess(abs(measured-expected),5*np.sqrt(expected*(1-expected)/hits.size))

    def test_loss_and_inverse_have_analytic_constant_stopping_limit(self):
        tables=constant_tables()
        energies=np.array([.1,1.,5.])
        remaining=tables.remove(energies,5e-5,'G4_Si')
        np.testing.assert_allclose(remaining,energies-.05,atol=1e-13)
        np.testing.assert_allclose(tables.restore(remaining,5e-5,'G4_Si'),energies,atol=1e-13)
        np.testing.assert_allclose(tables.materials['G4_Si']['light'](energies),.8*energies,atol=1e-13)
        self.assertEqual(tables.remove(.01,1e-3,'G4_B'),0.)

    def test_out_of_domain_correction_is_not_silently_clamped(self):
        tables=constant_tables()
        self.assertTrue(np.isnan(tables.restore(5.,.1,'G4_C')))
        with self.assertRaises(ValueError):tables.remove(11.,0.,'G4_C')
        with self.assertRaises(ValueError):tables.remove(1.,-.1,'G4_C')

    def test_layer_columns_include_density_angle_and_backing_side(self):
        response=r.Response()
        directions=np.tile(r.unit(r.FOIL_NORMAL),(1,3,1))
        detector=np.array([[0,1,0]])
        b,c,si=r.columns(directions,detector,np.array([.25]),response,constant_tables())
        np.testing.assert_allclose(b,12.5e-6*.25)
        np.testing.assert_allclose(c,14e-6)
        expected=np.array(response.dead_um)[detector]*1e-4*2/abs(np.sum(directions*r.NORMALS[detector],axis=-1))
        np.testing.assert_allclose(si,expected)

    def test_hit_patterns_partition_events_and_multiplicities_are_exclusive(self):
        probabilities=r.hit_pattern_probabilities(np.ones((1,3),bool),np.full((1,3),.5),'per_particle')
        np.testing.assert_allclose(probabilities,1/8)
        self.assertAlmostEqual(probabilities[0,[3,5,6]].sum(),3/8)
        self.assertAlmostEqual(probabilities[0,7],1/8)
        global_or=r.hit_pattern_probabilities(np.ones((1,3),bool),np.full((1,3),.5),'global_or')
        self.assertAlmostEqual(global_or[0,7],7/8)
        self.assertAlmostEqual(global_or[0,[3,5,6]].sum(),0)

    def test_adc_cut_and_trigger_midpoint_are_distinct(self):
        efficiency=r.trigger_efficiency(np.array([.01,.2,.35]),.2,.3)
        np.testing.assert_allclose(efficiency,[0,.5,1.])
        result=r.hit_pattern_probabilities(np.array([[True,True,False]]),np.ones((1,3)),'none')
        self.assertEqual(result[0,3],1.)

    def test_missing_particle_is_reconstructed_from_momentum(self):
        momentum=np.array([[[30.,0.,0.],[-10.,25.,0.],[-20.,-25.,0.]]])
        energies=np.sqrt(r.MASS*r.MASS+np.sum(momentum*momentum,axis=-1))
        kinetic=energies-r.MASS
        for missing in range(3):
            pattern=np.arange(3)!=missing
            got,p=r.reconstruct(kinetic,r.unit(momentum),energies.sum(axis=1),momentum.sum(axis=1),pattern)
            np.testing.assert_allclose(got,kinetic,atol=2e-12)
            np.testing.assert_allclose(p,momentum,atol=2e-10)
        self.assertTrue(r.triple_quality(momentum)[0])

    def test_dalitz_histogram_is_permutation_invariant_and_conserves_weight(self):
        energy=np.array([[1.,2.,3.],[2.,2.5,3.],[2.1,2.6,3.4]])
        weights=np.array([.5,1.,2.])
        edges=np.linspace(-.7,.7,141)
        original=r.symmetric_dalitz(energy,weights,edges)
        for permutation in r.PERMUTATIONS:
            np.testing.assert_allclose(r.symmetric_dalitz(energy[:,permutation],weights,edges),original,atol=1e-14)
        self.assertAlmostEqual(original.sum(),weights.sum())

    def test_unknown_trigger_model_and_invalid_width_fail(self):
        with self.assertRaises(ValueError):r.Response(trigger_mode='guessed')
        with self.assertRaises(ValueError):r.trigger_efficiency([.1],.1,0)


if __name__=='__main__':unittest.main()
