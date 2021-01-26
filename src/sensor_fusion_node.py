#!/usr/bin/python3

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

#Third Party
from pyUKF.ukf import UKF
import argparse
import yaml

# Python
import os
import math

#ROS
import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Pose, Twist, PoseWithCovariance, TwistWithCovariance, PoseArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler

#Sensor Fusion Node
from conversion import *
from utils import *

class SensorFusionNode:
    def __init__(self, args):

        port = rospy.get_param('config', args.config_file)
        stream = open(args.config_file, 'r')

        self.config = yaml.safe_load(stream)

        init_cov = gen_covariance_matrix(np.array(self.config["init_cov_sigmas"]))
        proc_noise = gen_covariance_matrix(np.array(self.config["proc_noise_sigmas"]))

        init_state = self.config["init_state"]
        self.filter_ = UKF(len(init_state), proc_noise, np.array(init_state), init_cov, args.alpha, args.k, args.beta, self.dynamics_func)

        # Flag indicating the filter has been initialized with an initial odometry estimate
        self.is_init_ = False

        rospy.init_node("sensor_fusion", anonymous=True)
        self.init_subscribers()

        self.fused_odom_pub_ = rospy.Publisher(args.out_odom_topic, Odometry, queue_size=10)
        print("Publishing state estimate on topic", args.out_odom_topic)
        self.sigmas_pub_ = rospy.Publisher(args.out_sigmas_topic, PoseArray, queue_size=10)
        print("Publishing UKF Sigmas on topic", args.out_sigmas_topic)

        self.rate_ = rospy.Rate(args.rate)
        self.prev_timestamp_ = rospy.Time.now()

        # Main filter loop
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            delta_t = (now - self.prev_timestamp_).to_sec()
            self.filter_.predict(delta_t)

            self.sigmas_pub_.publish(sigmas_2_pose_array(self.filter_.sigmas.T))
            self.fused_odom_pub_.publish(state_2_odom(StateVec(*self.filter_.get_state()), self.filter_.get_covar()))
            self.prev_timestamp_ = now
            self.rate_.sleep()

    def init_subscribers(self):
        self.subscribers_ = list()

        self.callbacks = {
        "init": self.ukf_init_callback,
        "imu" : self.ukf_imu_update,
        "odometry" : self.ukf_odom_update,
        "groundtruth" : self.ukf_gt_callback,
        }

        self.types = {
        "init": Odometry,
        "imu" : Imu,
        "odometry" : Odometry,
        "groundtruth" : Odometry,
        }

        for input in self.config["sensor_inputs"]:
            print("Listening for ", input["type"],"on topic", input["topic"])
            sub = rospy.Subscriber(input["topic"], self.types[input["type"]], self.callbacks[input["type"]], input["meas_covar"])
            self.subscribers_.append(sub)

    def ukf_init_callback(self, msg, data):
        """
        Callback for initializing the sensor fusion node with an init pose and velocity.
        If no base pose has been set, will set the current filter state
        to the input pose and velocity.
        """
        if not self.is_init_:
            state_vec = odom_to_state(msg)
            self.filter_.set_state(state_vec)
            self.is_init_ = True
            print("Initialized filter state to ",state_vec)

    def ukf_gt_callback(self, msg, data):
        """
        Callback for computing the residual of the current state vs groundtruth data.
        Note: Since this callback takes messages of type nav_msgs::Odometry,
        error is computed w.r.t. all states except acceleration.
        Prints the actual state, estimated state, diff, and error to the screen.
        """
        # TODO: Save these error values to a .txt or publish on a different topic.
        est_state = self.filter_.get_state()[:5]
        true_state = odom_to_state(msg)[:5]

        diff = true_state - est_state

        error = math.sqrt(diff.dot(diff))

        print("Actual: ", true_state)
        print("Est:    ", est_state)
        print("State Diff: ", diff)
        print("Dist Error: ", error)

    def ukf_imu_update(self, msg, meas_covar):
        """
        Callback for updating the sensor fusion node with a new IMU measurement.
        IMU measurements are used to update the angular yaw rate and acceleration in x.

        The measurement covariance takes the form
        [[cov(vyaw, vyaw), cov(vyaw, ax)],
          cov(ax, vyaw), cov(ax, ax)]

        If meas_covar is not provided in the yaml file, this callback will extract the covariance
        from the message itself and put it in the above form.
        """

        vyaw = msg.angular_velocity.z
        ax = msg.linear_acceleration.x

        if meas_covar is None:
            r_ang_vel = np.reshape(msg.angular_velocity_covariance, (3, 3))
            r_lin_acc = np.reshape(msg.linear_acceleration_covariance, (3, 3))
            meas_covar = [[r_ang_vel[2, 2], 0], [0, r_lin_acc[0,0]]]

        states_to_update = [ivyaw, iax]
        measurements = [vyaw, ax]
        self.filter_.update(states_to_update, measurements, meas_covar)

    def ukf_odom_update(self, msg, meas_covar):
        """
        Callback for updating the sensor fusion node with a new odometry measurement.
        Updates the state vector with new velocity_x and velocity_yaw.

        Odometry measurements are used to update the angular yaw rate and velocity in x.

        The measurement covariance takes the form
        [[cov(vx, vx), cov(vx, vyaw)],
         cov(vyaw, vx), cov(vyaw, vyaw)]

        If meas_covar is not provided in the yaml file, this callback will extract the covariance
        from the message itself and put it in the above form.
        """

        if meas_covar is None:
            r = np.reshape(msg.twist.covariance, (3, 3))
            meas_covar = [[r[0,0], r[0,5]],[r[5,0], r[5,5]]]

        vx = msg.twist.twist.linear.x
        vyaw = msg.twist.twist.angular.z

        self.filter_.update([ivx, ivyaw], [vx, vyaw], meas_covar)


    def dynamics_func(self, state_vec, timestep, inputs):
        """
        Dynamics function describing how the state is affected without input.

        As part of our state vector, we are tracking
        [x, y, yaw, vx, vyaw, ax]

        where
        - (x, y) is the absolute cartesian robot pose (meters)
        - yaw is the absolute heading (radians)
        - vx is the velocity along the x direction (meters/s)
        - vw is the angular yaw rate (radians/s)
        - ax is the acceleration along the x direction (meters/s^2)

        (x, y, yaw) is indirectly estimated from
        the velocity and acceleration measurements given from the IMU and odometry,
        which the absolute pose is integrated from w.r.t. time.
        """

        state = StateVec(*state_vec)
        new_state = state
        new_state.x += timestep * state.vx * math.cos(state.yaw)
        new_state.y += timestep * state.vx * math.sin(state.yaw)
        new_state.yaw += timestep * state.vyaw
        new_state.vx += timestep * state.ax

        return np.array(new_state)


def main():
    parser = argparse.ArgumentParser(description='ROS node to estimate fused odometry from multiple odometry and IMU inputs.')

    # Node config parameters
    parser.add_argument('--rate',action='store', type = float, default=10, help = 'filter rate')
    parser.add_argument('--config_file',action='store', type = str, default=os.getcwd() + "/src/sensor_fusion_node/config/fusion_config.yaml", help = 'config file' )
    parser.add_argument('--out_odom_topic',action='store', type = str, default="fused_odom", help = 'Output fused odom topic' )
    parser.add_argument('--out_sigmas_topic',action='store', type = str, default="sigmas", help = 'Output sigma points topic' )

    # Filter config parameters
    parser.add_argument('--alpha',action='store', type = float, default= 0.04, help = 'Tuning parameter - sigma point spread' )
    parser.add_argument('--k',action='store', type = float, default=0.0, help = 'Tuning parameter - typically either 0 or (3 - n_states)' )
    parser.add_argument('--beta',action='store', type = float, default=2.0, help = 'Tuning parameter - typically 2 is ideal for gaussian distributions' )

    args, _ = parser.parse_known_args()
    np.set_printoptions(precision=2, suppress=True, linewidth=200)

    try:
        node = SensorFusionNode(args)
    except rospy.ROSInterruptException:
        print ("Shutting down...")

if __name__ == "__main__":
    main()
