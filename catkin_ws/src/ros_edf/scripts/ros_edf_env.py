#!/usr/bin/env python

import rospy

from edf_env.env import UR5Env, MugEnv
from edf_env.ros_wrapper import UR5EnvRosWrapper

def run():
    env = MugEnv(use_gui=True)
    env_ros = UR5EnvRosWrapper(env=env, monitor_refresh_rate=0)
    rospy.spin()


if __name__ == '__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass