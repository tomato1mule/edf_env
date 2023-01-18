import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from edf_env.env import UR5Env

class UR5EnvRosWrapper():
    def __init__(self, env: UR5Env):
        self.env = env
        self.movable_joints_id = []
        for id, joint_type in enumerate(self.env.robot_joint_type_list):
            if joint_type != 'JOINT_FIXED':
                self.movable_joints_id.append(id)

        self.joint_pub = rospy.Publisher('joint_states', JointState, latch=False)
        rospy.init_node('edf_env', anonymous=True)

        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            self.publish_joint_info()
            rate.sleep()


    def publish_joint_info(self):
        pos, vel = self.env.get_joint_states(list(range(self.env.n_joints)))

        header = Header()
        header.stamp = rospy.Time.now()

        # for id in self.movable_joints_id:
        #     msg = JointState()
        #     msg.header = header
        #     msg.name = self.env.robot_joint_name_list[id]
        #     msg.position = [pos[id]]
        #     msg.velocity = [vel[id]]
        #     self.joint_pub.publish(msg)
        msg = JointState()
        msg.header = header
        for id in self.movable_joints_id:
            msg.name.append(self.env.robot_joint_name_list[id])
            msg.position.append(pos[id])
            msg.velocity.append(vel[id])
        self.joint_pub.publish(msg)
        

