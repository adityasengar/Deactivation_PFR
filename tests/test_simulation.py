import unittest
import numpy as np
import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pfr_simulation import d_propylene_conc_dz, update_activity

class TestSimulation(unittest.TestCase):

    def setUp(self):
        self.params = {
            "k_reaction": 5.0,
            "xi_m": 2.0,
            "initial_x": 1.0,
            "k_d": 1.0,
            "dt": 0.1
        }

    def test_d_propylene_conc_dz(self):
        # Test with some values
        activity = 1.0
        propylene_conc = 1.0
        
        # dx/dz' = -k_reaction * (x*λ₁ + x²*λ₁*ξ_m / x₀)
        # dx/dz' = -5.0 * (1*1 + 1^2*1*2 / 1) = -5.0 * (1 + 2) = -15.0
        expected_dx_dz = -15.0
        
        dx_dz = d_propylene_conc_dz(activity, propylene_conc, self.params)
        self.assertAlmostEqual(dx_dz, expected_dx_dz)

    def test_update_activity(self):
        # Test with some values
        activity_profile = np.array([1.0, 0.8, 0.6])
        propylene_profile = np.array([1.0, 0.5, 0.2])
        
        # d(lambda)/dt = -k_d * lambda * x
        # d_lambda_dt = -1.0 * [1.0*1.0, 0.8*0.5, 0.6*0.2] = [-1.0, -0.4, -0.12]
        # new_activity = old_activity + d_lambda_dt * dt
        # new_activity = [1.0, 0.8, 0.6] + [-1.0, -0.4, -0.12] * 0.1
        # new_activity = [1.0, 0.8, 0.6] + [-0.1, -0.04, -0.012]
        # new_activity = [0.9, 0.76, 0.588]
        
        expected_new_activity = np.array([0.9, 0.76, 0.588])
        new_activity = update_activity(activity_profile, propylene_profile, self.params)
        np.testing.assert_array_almost_equal(new_activity, expected_new_activity)

if __name__ == '__main__':
    unittest.main()
