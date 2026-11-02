"""Recorded-state compatibility must handle absent features and reject physical drift."""
import unittest
import mujoco
from model_identity import model_fingerprint, verify_model


class ModelIdentityTests(unittest.TestCase):
    def model(self):
        return mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="2"/>'
            '</body></worldbody></mujoco>')

    def test_absent_mocap_skin_and_sensor_arrays_remain_compatible(self):
        expected = model_fingerprint(self.model())
        verify_model(self.model(), expected)
        self.assertEqual(expected['dimensions']['nmocap'], 0)
        self.assertEqual(expected['dimensions']['nsensor'], 0)

    def test_changed_mass_and_timestep_are_rejected(self):
        model = self.model()
        expected = model_fingerprint(model)
        model.body_mass[1] += .01
        with self.assertRaisesRegex(ValueError, 'arrays.body_mass'):
            verify_model(model, expected)
        model = self.model()
        model.opt.timestep *= 2
        with self.assertRaisesRegex(ValueError, 'options.timestep'):
            verify_model(model, expected)


if __name__ == '__main__':
    unittest.main()
