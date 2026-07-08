# 方式 1: 和原始流程一致的入口（推荐）
roslaunch iden_controller cruise_navfn.launch
rosrun iden_controller process_navfn.py

# 小小的动态避障
roslaunch iden_controller cruise_navfn_v2_wide.launch
rosrun iden_controller cruise_demo.py

# 小小的第一部分多定点成功导航
 roslaunch iden_controller cruise_navfn_success1.launch
 rosrun iden_controller process_navfn_success1.py

roslaunch iden_controller cruise_navfn_success2.launch
 rosrun iden_controller process_navfn_success2.py

# 目前最好的第一部分导航
roslaunch iden_controller global_first_graph_nav.launch

# 第一部分仅导航加扫码
roslaunch iden_controller global_first_graph_nav_qr_room.launch

# 第一部分所有
roslaunch iden_controller subtask1_real_factory.launch

~/instant_ws/src/iden_controller/scripts/start_subtask1_real_factory_route.sh