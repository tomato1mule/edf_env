import rospy
from sensor_msgs.msg import JointState

class RosRobotInterface():
    def __init__(self):
        self.joint_pub = rospy.Publisher('joint_states', JointState, queue_size=1, latch=True)
        