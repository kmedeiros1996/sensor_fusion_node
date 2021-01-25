UKF-based rospy node to estimate a fused odometric output from multiple odometry inputs.

The YAML file provided allows specification for any number of sensor inputs. Inputs can be of type nav_msgs::Odometry or sensor_msgs::Imu. There is also the option to provide groundtruth data to compare the current state estimate against. The provided yaml file has several example odometry topics. 
