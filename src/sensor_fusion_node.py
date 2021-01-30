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

#Sensor Fusion Node
from conversion import *
from utils import *
from data_receivers import *


class SensorFusionNode:
    def __init__(self, args):

        file = rospy.get_param('config', args.config_file)

        config_file = open(args.config_file, 'r')
        self.config = yaml.safe_load(config_file)
        config_file.close()

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

        self.spin()

    def spin(self):
        """
        Main filter loop.
        For each sensor provided in the yaml, will update the filter each iteration
        if the measurement is available.
        """
        try:
            while not rospy.is_shutdown():
                now = rospy.Time.now()
                delta_t = (now - self.prev_timestamp_).to_sec()
                self.filter_.predict(delta_t)

                if not self.is_init_:
                    if "init_pose" not in self.config.keys():
                        self.is_init_ = True
                    else:
                        init_state = self.init_receiver_.get_data()
                        if init_state is not None:
                            self.filter_.set_state(init_state)
                            print("Initialized filter state to", init_state)
                            self.is_init_ = True

                if self.is_init_:
                    for topic, sensor in self.sensor_receivers_.items():
                        measurement = sensor.get_data()
                        if measurement is not None:
                            self.filter_.update(sensor.update_indices, measurement, sensor.covariance)

                if "groundtruth" in self.config.keys():
                    true_state = self.gt_receiver_.get_data()
                    if true_state is not None:
                        est_state = self.filter_.get_state()[:5]
                        diff = true_state - est_state
                        error_dist = math.sqrt(diff[:2].dot(diff[:2]))

                        if diff[2] < -math.pi: diff[2]+=2.0*math.pi
                        if diff[2] >= math.pi: diff[2]-=2.0*math.pi

                        print("Actual: ", true_state)
                        print("Est:    ", est_state)
                        print("State Diff: ", diff)
                        print("Distance XY: ", error_dist)
                        print("Error Yaw  : ", np.degrees(diff[2]))

                self.sigmas_pub_.publish(sigmas_2_pose_array(self.filter_.sigmas.T, self.config["fixed_frame"]))
                self.fused_odom_pub_.publish(state_2_odom(StateVec(*self.filter_.get_state()), self.filter_.get_covar(), self.config["fixed_frame"]))
                self.prev_timestamp_ = now
                self.rate_.sleep()

        except rospy.ROSInterruptException as e:
            print ("Shutting down...", e)

    def init_subscribers(self):
        """
        Instantiates data receiver objects to read measurements for the filter to update.
        """
        self.sensor_receivers_ = dict()

        receivers = {
        "imu" : IMUReceiver,
        "odometry" : OdometryReceiver
        }

        if "groundtruth" in self.config.keys():
            gt_data = self.config["groundtruth"]
            self.gt_receiver_ = GroundtruthReceiver(gt_data["topic"], self.config["fixed_frame"], gt_data["frame"])

        if "init_pose" in self.config.keys():
            init_data = self.config["init_pose"]
            self.init_receiver_ = InitReceiver(init_data["topic"], self.config["fixed_frame"], init_data["frame"])

        for input in self.config["sensor_inputs"]:
            if input["enable"]:
                self.sensor_receivers_[input["topic"]] = receivers[input["type"]](input["topic"], self.config["fixed_frame"], input["frame"], input["meas_covar"])

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
    parser.add_argument('--rate',action='store', type = float, default=200, help = 'filter rate')
    parser.add_argument('--config_file',action='store', type = str, default=os.getcwd() + "/src/sensor_fusion_node/config/fusion_config.yaml", help = 'config file' )
    parser.add_argument('--out_odom_topic',action='store', type = str, default="fused_odom", help = 'Output fused odom topic' )
    parser.add_argument('--out_sigmas_topic',action='store', type = str, default="sigmas", help = 'Output sigma points topic' )

    # Filter config parameters
    parser.add_argument('--alpha',action='store', type = float, default= 0.04, help = 'Tuning parameter - sigma point spread' )
    parser.add_argument('--k',action='store', type = float, default=0.0, help = 'Tuning parameter - typically either 0 or (3 - n_states)' )
    parser.add_argument('--beta',action='store', type = float, default=2.0, help = 'Tuning parameter - typically 2 is ideal for gaussian distributions' )

    args, _ = parser.parse_known_args()
    np.set_printoptions(precision=2, suppress=True, linewidth=200)

    node = SensorFusionNode(args)

if __name__ == "__main__":
    main()
