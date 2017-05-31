#! /usr/bin/env sh


rosbag record  \
	/app/camera/rgb/image_raw/compressed \
	/depth/image_raw/compressed \
	/depth/camera_info \
	/depth/points \
	/ir/image_raw/compressed \
	/manual_control/lights \
	/manual_control/speed \
	/manual_control/steering \
	/manual_control/stop_start \
	/model_car/yaw \
	/motor_control/twist \
	/odom
