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

from threading import Lock
import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import tf2_ros

# Sensor Fusion Node
from conversion import *

class DataReceiver:
    """
    Asynchronous data receiver encapsulating a subscriber and a TF buffer.
    Updates data with latest received message.
    When data is pulled from the receiver, sets the data_available flag to false.
    """
    def __init__(self, topic, filter_frame, sensor_frame, type, callback):
        self.data = None
        self.lock = Lock()
        self.data_available = False

        self.topic = topic
        self.sensor_frame = sensor_frame
        self.filter_frame = filter_frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.sub = rospy.Subscriber(self.topic, type, callback)
        print("Listening for ",type," on topic",self.topic,"with sensor frame",self.sensor_frame)

    # TODO: Use the following method to transform messages into the filter frame
    def get_transform(self):
        try:
            trans = self.tf_buffer.lookup_transform(self.filter_frame, self.sensor_frame, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            trans = None

        return trans

    def put_data(self, data):
        with self.lock:
            self.data = data
            self.data_available = True

    def get_data(self):
        with self.lock:
            if self.data_available:
                self.data_available = False
                return self.data
            return None


class IMUReceiver(DataReceiver):
    def __init__(self, topic, filter_frame, sensor_frame, covariance=None):
        super().__init__(topic, filter_frame, sensor_frame, Imu, self.callback)
        self.covariance = covariance
        self.update_indices = (ivyaw, iax, iay)

    def callback(self, msg):
        """
        Callback for providing a new IMU measurement to the filter.
        IMU measurements are used to update the angular yaw rate and acceleration in x.

        The measurement covariance takes the form
        [[cov(vyaw, vyaw),  0, 0],
         [0,   cov(ax, ax),    0],
          0,   0,    cov(ay, ay)]

        If meas_covar is not provided in the yaml file, this callback will extract the covariance
        from the message itself and put it in the above form.
        """

        vyaw = msg.angular_velocity.z
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y

        if self.covariance is None:
            r_ang_vel = np.reshape(msg.angular_velocity_covariance, (3, 3))
            r_lin_acc = np.reshape(msg.linear_acceleration_covariance, (3, 3))
            self.covariance = [[r_ang_vel[2, 2], 0, 0], [0, r_lin_acc[0,0], 0], [0, 0, r_lin_acc[1, 1]]]

        self.put_data([vyaw, ax, ay])


class OdometryReceiver(DataReceiver):
    def __init__(self, topic, filter_frame, sensor_frame, covariance=None):
        super().__init__(topic, filter_frame, sensor_frame, Odometry, self.callback)
        self.covariance = covariance
        self.update_indices = (ivx, ivy, ivyaw)

    def callback(self, msg):
        """
        Callback for providing a new odometry measurement to the filter.
        Updates the state vector with new velocity_x and velocity_yaw.

        Odometry measurements are used to update the angular yaw rate and velocity in x.

        The measurement covariance takes the form
        [[cov(vx, vx),  cov(vx, vy), cov(vx, vyaw)],
         [cov(vy, vx),  cov(vy, vy), cov(vy, vyaw)],
         cov(vyaw, vx), cov(vx, vy),  cov(vyaw, vyaw)]

        If meas_covar is not provided in the yaml file, this callback will extract the covariance
        from the message itself and put it in the above form.
        """

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vyaw = msg.twist.twist.angular.z

        if self.covariance is None:
            r = np.reshape(msg.twist.covariance, (3, 3))
            self.covariance = [[r[0,0], r[0,1], r[0,5]],[r[1,0], r[1,1], r[1,5]], [r[5,0], r[5,1], r[5,5]]]

        self.put_data([vx, vy, vyaw])


class GroundtruthReceiver(DataReceiver):
    def __init__(self, topic, filter_frame, sensor_frame):
        super().__init__(topic, filter_frame, sensor_frame, Odometry, self.callback)

    def callback(self, msg):
        state_vec = odom_to_state(msg)
        self.put_data(state_vec[:5])

class InitReceiver(DataReceiver):
    def __init__(self, topic, filter_frame, sensor_frame):
        super().__init__(topic, filter_frame, sensor_frame, Odometry, self.callback)

    def callback(self, msg):
        state_vec = odom_to_state(msg)
        self.put_data(state_vec)
