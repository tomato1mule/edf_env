#!/usr/bin/env python

import rospy

from edf_env.env import UR5Env
from edf_env.ros_wrapper import UR5EnvRosWrapper

def run():
    env = UR5Env(use_gui=False)
    env_ros = UR5EnvRosWrapper(env=env)
    rospy.spin()


if __name__ == '__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass