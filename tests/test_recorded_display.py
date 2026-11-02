"""Recorded display restoration must not hide a physical model mismatch."""
import unittest
import mujoco
from model_identity import model_fingerprint, restore_recorded_display, verify_model


class RecordedDisplayTests(unittest.TestCase):
    def fixture(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="vessel">
          <freejoint/><geom name="solid" type="sphere" size=".1" mass="2"/>
          <geom name="liquid" type="sphere" size=".05" pos="0 0 .02"
                contype="0" conaffinity="0" density="0"/>
        </body></worldbody></mujoco>''')
        index = model.geom('liquid').id
        state = {'schema_version': 1, 'geoms': [{'name': 'liquid', 'body': 'vessel',
                 'geom_pos': model.geom_pos[index].tolist(),
                 'geom_size': model.geom_size[index].tolist()}]}
        return model, index, state

    def test_recorded_display_state_recovers_exact_model_identity(self):
        model, index, state = self.fixture()
        expected = model_fingerprint(model)
        model.geom_pos[index, 2] += .01
        model.geom_size[index, 0] *= .5
        self.assertNotEqual(model_fingerprint(model), expected)
        self.assertEqual(restore_recorded_display(model, state), 1)
        self.assertEqual(model_fingerprint(model), expected)

    def test_contact_enabled_display_is_rejected(self):
        model, index, state = self.fixture()
        model.geom_contype[index] = 1
        with self.assertRaisesRegex(ValueError, 'collision geometry'):
            restore_recorded_display(model, state)

    def test_mass_change_remains_rejected_after_display_restoration(self):
        model, index, state = self.fixture()
        expected = model_fingerprint(model)
        model.geom_pos[index, 2] += .01
        model.body_mass[model.body('vessel').id] *= 2
        restore_recorded_display(model, state)
        with self.assertRaisesRegex(ValueError, 'arrays.body_mass'):
            verify_model(model, expected)

    def test_solid_geometry_change_is_not_restored(self):
        model, _, state = self.fixture()
        expected = model_fingerprint(model)
        model.geom_size[model.geom('solid').id, 0] *= 2
        restore_recorded_display(model, state)
        with self.assertRaisesRegex(ValueError, 'arrays.geom_size'):
            verify_model(model, expected)


if __name__ == '__main__':
    unittest.main()
