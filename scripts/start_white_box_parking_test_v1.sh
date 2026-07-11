#!/usr/bin/env bash
set -e

source "$HOME/instant_ws/devel/setup.bash"
exec roslaunch iden_controller white_box_parking_test_v1.launch "$@"
