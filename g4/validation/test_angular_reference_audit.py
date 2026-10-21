"""Independent mathematical and frame checks for the angular-reference audit."""

import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss

import audit_162_angular_reference as audit


class AngularReferenceTests(unittest.TestCase):
    def test_cg_complete_orthogonal_basis(self):
        # Test full coupled bases, not only the five coefficients used by the plot.
        for total_m in range(-4, 5):
            pairs = [(m1, total_m-m1) for m1 in range(-2, 3) if abs(total_m-m1) <= 2]
            angular_momenta = range(abs(total_m), 5)
            matrix = np.array([[audit.clebsch_gordan(2, m1, 2, m2, j, total_m)
                                for j in angular_momenta] for m1, m2 in pairs])
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(len(pairs)), atol=2e-14)

    def test_cg_independent_j_squared_diagonalization(self):
        for primary_l in (0, 2, 4):
            for m in range(-2, 3):
                value = audit.clebsch_gordan(primary_l, 0, 2, m, 2, m)**2
                self.assertAlmostEqual(value, audit.cg_squared_from_j2(primary_l, m), places=13)

    def test_cg_selection_rules(self):
        self.assertEqual(audit.clebsch_gordan(2, 0, 2, 1, 2, 0), 0)
        self.assertEqual(audit.clebsch_gordan(2, 3, 2, -1, 2, 2), 0)
        self.assertEqual(audit.clebsch_gordan(2, 0, 2, 0, 5, 0), 0)

    def test_coupled_amplitude_integrates_to_one(self):
        # Integrate both spherical directions directly, using different azimuth
        # grids for the two particles. No closed-form angular law is used.
        x, w = leggauss(6)
        phi = np.arange(12) * 2*np.pi/12
        xx, pp = np.meshgrid(x, phi, indexing="ij")
        vectors = np.column_stack((np.sqrt(1-xx.ravel()**2)*np.cos(pp.ravel()),
                                   np.sqrt(1-xx.ravel()**2)*np.sin(pp.ravel()), xx.ravel()))
        weights = np.repeat(w, len(phi)) * 2*np.pi/len(phi)
        first = np.repeat(vectors, len(vectors), axis=0)
        second = np.tile(vectors, (len(vectors), 1))
        integral = np.sum(audit.joint_angular_density(first, second)*np.outer(weights, weights).ravel())
        self.assertAlmostEqual(integral, 1, places=12)

    def test_coupled_amplitude_rotational_invariance(self):
        rng = np.random.default_rng(16219)
        p, s = rng.normal(size=(2, 91, 3))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotation[:, 0] *= np.linalg.det(rotation)
        np.testing.assert_allclose(audit.joint_angular_density(p, s),
                                   audit.joint_angular_density(p @ rotation, s @ rotation),
                                   atol=2e-15, rtol=2e-13)

    def test_general_amplitude_relative_angle(self):
        rng = np.random.default_rng(16220)
        p, s = rng.normal(size=(2, 113, 3))
        x = np.sum(p*s, axis=1)/np.linalg.norm(p, axis=1)/np.linalg.norm(s, axis=1)
        np.testing.assert_allclose(audit.joint_angular_density(p, s)*8*np.pi**2,
                                   audit.single_path_l2_pdf(x), atol=3e-14)

    def test_pure_s_wave_isotropic(self):
        x = np.linspace(-1, 1, 71)
        np.testing.assert_allclose(audit.pdf_from_coupling(x, primary_l=0), 0.5, atol=3e-14)

    def test_l2_coupling_matches_closed_form(self):
        x = np.linspace(-1, 1, 211)
        for method in ("racah", "eigensolver"):
            np.testing.assert_allclose(audit.pdf_from_coupling(x, method=method),
                                       audit.single_path_l2_pdf(x), atol=3e-14)

    def test_pdf_measure_and_legendre_moments(self):
        x, w = leggauss(32)
        for fn, a2, a4 in ((audit.legendre_pdf, -0.489, 0.647),
                           (audit.single_path_l2_pdf, -15/49, 36/49),
                           (audit.published_eq2_pdf, 40/203, -96/203)):
            y = fn(x)
            self.assertTrue(np.all(y > 0))
            self.assertAlmostEqual(float(w @ y), 1, places=12)
            self.assertAlmostEqual(float(w @ (y*(3*x*x-1)/2))*5, a2, places=12)
            self.assertAlmostEqual(float(w @ (y*(35*x**4-30*x*x+3)/8))*9, a4, places=12)

    def test_axis_reversal_does_not_change_even_law(self):
        x = np.linspace(-1, 1, 71)
        for fn in (audit.legendre_pdf, audit.single_path_l2_pdf, audit.published_eq2_pdf):
            np.testing.assert_allclose(fn(x), fn(-x), atol=1e-15)

    def test_shifted_plane_intensity_is_periodic_and_not_polar_pdf(self):
        theta = np.linspace(0, np.pi, 10001)
        w = audit.planar_weight(theta, -12)
        np.testing.assert_allclose(w, audit.planar_weight(theta+np.pi, -12), atol=3e-14)
        mean = np.trapezoid(w, theta)/np.pi
        self.assertAlmostEqual(float(mean), 1-0.489/4+0.647*9/64, places=11)

    def test_boost_round_trip(self):
        rng = np.random.default_rng(16221)
        p = rng.normal(size=(100, 3))*50
        e = np.sqrt(np.sum(p*p, axis=1)+audit.ALPHA_MASS_MEV**2)
        beta = rng.normal(size=(100, 3))*0.005
        ep, pp = audit._rest_frame(e, p, beta)
        e2, p2 = audit._rest_frame(ep, pp, -beta)
        np.testing.assert_allclose(e, e2, atol=1e-10, rtol=0)
        np.testing.assert_allclose(p, p2, atol=1e-10, rtol=0)

    def test_in_plane_azimuth_definition(self):
        primary = np.array([1.0, 0, 0])
        beam = np.array([0, 0, 1.0])
        normal = np.cross(beam, primary)
        in_plane = np.cross(normal, primary)
        for phi in (-2.3, -1.1, 0, 0.9, 2.1):
            v = 0.6*primary+0.8*(np.cos(phi)*in_plane+np.sin(phi)*normal)
            reconstructed = np.arctan2(v @ normal, v @ in_plane)
            self.assertAlmostEqual(reconstructed, phi)


if __name__ == "__main__":
    unittest.main()
