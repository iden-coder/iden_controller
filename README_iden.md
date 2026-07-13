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

# 有大模型的第一部分所有
~/instant_ws/src/iden_controller/scripts/start_subtask1_real_factory_route.sh

# 图片文字识别
roslaunch iden_controller factory_ocr_from_package.launch

# 第一二部分所有（还没有完善）
~/instant_ws/src/iden_controller/scripts/start_subtask1_factory_delivery_complete_center_only_front_first_v1.sh

# 测试停止在白框里的能力（白框法）
~/instant_ws/src/iden_controller/scripts/start_white_box_parking_test_v7.sh

# 测试停止在白框里的能力（中心线法）
~/instant_ws/src/iden_controller/scripts/start_factory_sign_center_only_parking_test_v2.sh

# 测试停止在白框里的能力（融合法）
~/instant_ws/src/iden_controller/scripts/start_factory_sign_center_parking_test_v4.sh