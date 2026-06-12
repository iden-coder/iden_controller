# 方式 1: 和原始流程一致的入口（推荐）
roslaunch iden_controller cruise_navfn.launch
rosrun iden_controller process_navfn.py

roslaunch iden_controller cruise_navfn_v2_wide.launch
rosrun iden_controller cruise_demo.py
