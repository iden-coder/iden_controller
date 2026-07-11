#!/usr/bin/env bash
set -e
source /opt/ros/noetic/setup.bash
source /home/ucar/instant_ws/devel/setup.bash
exec roslaunch iden_controller white_box_parking_test_v7.launch "$@"
