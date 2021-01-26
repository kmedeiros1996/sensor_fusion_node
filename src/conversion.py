'''Copyright 2021 Kyle M. Medeiros

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from recordclass import recordclass
import numpy as np

import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Pose, Twist, PoseWithCovariance, TwistWithCovariance, PoseArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler

ix, iy, iyaw, ivx, ivyaw, iax = np.arange(0, 6)

StateVec = recordclass("StateVec", "x y yaw vx vyaw ax")

def state_2_pose(state_vec):
    out = Pose()
    out.position.x = state_vec.x
    out.position.y = state_vec.y
    out.position.z = 0
    out_q = quaternion_from_euler(0, 0, state_vec.yaw)
    out.orientation = Quaternion(*out_q)
    return out

def state_2_twist(state_vec):
    out = Twist()
    out.linear.x = state_vec.vx
    out.linear.y = 0
    out.linear.z = 0
    out.angular.x = 0
    out.angular.y = 0
    out.angular.z = state_vec.vyaw
    return out

def fill_pose_covariance(cov):
    out = np.eye(6)
    out[:iy+1, :iy+1] = cov[:iy+1, :iy+1]

    out[:2, 5] = cov[:2, 2]
    out[5, :2] = out[:2, 5]

    out[5, 5] = cov[2, 2]

    return out.flatten()

def fill_twist_covariance(cov):
    out = np.eye(6)
    out[0, 0] = cov[3, 3]
    out[0, 5] = cov[2, 3]
    out[5, 0] = out[0, 5]
    out[5, 5] = cov[4, 4]
    return out.flatten()

def state_to_twist_with_cov(state_vec, cov):
    out = TwistWithCovariance()
    out.twist = state_2_twist(state_vec)
    out.covariance = fill_twist_covariance(cov)
    return out

def state_to_pose_with_cov(state_vec, cov):
    out = PoseWithCovariance()
    out.pose = state_2_pose(state_vec)
    out.covariance = fill_pose_covariance(cov)
    return out

def state_2_odom(state_vec, covariance):
    out = Odometry()
    out.header.stamp = rospy.Time.now()
    out.header.frame_id = "odom"
    out.pose = state_to_pose_with_cov(state_vec, covariance)
    out.twist = state_to_twist_with_cov(state_vec, covariance)
    return out

def sigmas_2_pose_array(sigmas):
    out = PoseArray()
    out.header.stamp = rospy.Time.now()
    out.header.frame_id = "odom"
    for sig_pt in sigmas:
        out.poses.append(state_2_pose(StateVec(*sig_pt)))
    return out

def odom_to_state(msg):
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y

    q = msg.pose.pose.orientation
    _,_,yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))

    vx = msg.twist.twist.linear.x
    vyaw = msg.twist.twist.angular.z

    return np.array([x, y, yaw, vx, vyaw, 0])
