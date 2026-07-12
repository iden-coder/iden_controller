#!/usr/bin/env bash
set -e
source /opt/ros/noetic/setup.bash
source /home/ucar/instant_ws/devel/setup.bash
exec roslaunch iden_controller factory_sign_center_parking_test_v3.launch "$@"
