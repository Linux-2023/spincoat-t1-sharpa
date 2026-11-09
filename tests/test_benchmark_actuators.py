'''Robot action ownership follows model topology, not misleading names.'''
import unittest

import mujoco

from benchmark.native_sources import robot_actuators


class ActuatorOwnershipTests(unittest.TestCase):
    def test_scene_joint_with_robot_prefix_cannot_be_controlled(self):
        model = mujoco.MjModel.from_xml_string('''
        <mujoco>
          <worldbody>
            <body name='chassis_base'>
              <body name='wheel'>
                <joint name='wheel_RF_Joint'/>
                <geom type='sphere' size='.02' mass='.1'/>
              </body>
            </body>
            <body name='laboratory_object' pos='1 0 0'>
              <joint name='astribot_fake_joint'/>
              <geom type='sphere' size='.02' mass='.1'/>
            </body>
          </worldbody>
          <actuator>
            <velocity name='astribot_chassis_motor_0' joint='wheel_RF_Joint' kv='1'/>
            <position name='astribot_impostor' joint='astribot_fake_joint' kp='1'/>
          </actuator>
        </mujoco>''')
        with self.assertRaises(ValueError):
            robot_actuators(model)


if __name__ == '__main__':
    unittest.main()
