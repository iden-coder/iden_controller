#ifndef BASE_DRIVER_H_
#define BASE_DRIVER_H_

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Joy.h>
#include <iostream>
#include <serial/serial.h>  //ROS的串口包 http://wjwwood.io/serial/doc/1.1.0/index.html
#include <math.h>
#include <fstream>
#include <iden_controller/data_struct_iden.h>
#include <iden_controller/GetMaxVel.h>
#include <iden_controller/SetMaxVel.h>
#include <sensor_msgs/BatteryState.h>
#include <iden_controller/GetBatteryInfo.h>
#include <iden_controller/SetLEDMode.h>
#include <boost/thread.hpp>
#include <string>
#include <std_srvs/Empty.h>
#include <std_srvs/Trigger.h>
#include <std_msgs/UInt8.h>
#include <std_msgs/UInt16.h>
#include <std_msgs/Float64MultiArray.h>
#include <ros/package.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Pose2D.h>
#include <iden_controller/fdilink_data_struct_iden.h>
#include <iden_controller/crc_table_iden.h>

using namespace std;
#define ODOM_POSE_COVARIANCE {1e-3, 0, 0, 0, 0, 0,\
                              0, 1e-3, 0, 0, 0, 0,\
                              0, 0, 1e6, 0, 0, 0,\
                              0, 0, 0, 1e6, 0, 0,\
                              0, 0, 0, 0, 1e6, 0,\
                              0, 0, 0, 0, 0, 1e3}

#define ODOM_POSE_COVARIANCE2 {1e-9, 0, 0, 0, 0, 0,\
                              0, 1e-3, 1e-9, 0, 0, 0,\
                              0, 0, 1e6, 0, 0, 0,\
                              0, 0, 0, 1e6, 0, 0,\
                              0, 0, 0, 0, 1e6, 0,\
                              0, 0, 0, 0, 0, 1e-9}

#define ODOM_TWIST_COVARIANCE {1e-3, 0, 0, 0, 0, 0,\
                               0, 1e-3, 0, 0, 0, 0,\
                               0, 0, 1e6, 0, 0, 0,\
                               0, 0, 0, 1e6, 0, 0,\
                               0, 0, 0, 0, 1e6, 0,\
                               0, 0, 0, 0, 0, 1e3}

#define ODOM_TWIST_COVARIANCE2 {1e-9, 0, 0, 0, 0, 0,\
                                0, 1e-3, 1e-9, 0, 0, 0,\
                                0, 0, 1e6, 0, 0, 0,\
                                0, 0, 0, 1e6, 0, 0,\
                                0, 0, 0, 0, 1e6, 0,\
                                0, 0, 0, 0, 0, 1e-9}

namespace ucarController
{
#define Pi 3.1415926

#define WRITE_DATA_LONGTH 8
#define READ_MSG_LONGTH  14 // 13+1(new)
#define READ_DATA_LONGTH 12 // 11+1(new)
#define WRITE_MSG_LONGTH 16 // 13+3(new)
#define CS_LONGTH 1

//LED_MODE
#define LED_MODE_NORMAL  0
#define LED_MODE_BLINK   1
#define LED_MODE_BREATH  2

//MOTOR_MODE
#define MOTOR_MODE_JOY     0
#define MOTOR_MODE_CMD     1
#define MOTOR_MODE_MOVE    2

class baseBringup
{
public:
  baseBringup();
  ~baseBringup();
  void velCallback(const geometry_msgs::Twist::ConstPtr& msg);
  void joyCallback(const sensor_msgs::Joy::ConstPtr& msg);
  bool getMaxVelCB(iden_controller::GetMaxVel::Request &req, iden_controller::GetMaxVel::Response &res);
  bool setMaxVelCB(iden_controller::SetMaxVel::Request &req, iden_controller::SetMaxVel::Response &res);
  bool stopMoveCB (std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);
  bool getBatteryStateCB(iden_controller::GetBatteryInfo::Request &req, iden_controller::GetBatteryInfo::Response &res);
  bool setLEDCallBack(iden_controller::SetLEDMode::Request &req, iden_controller::SetLEDMode::Response &res);
  bool updateMileage(double vx, double vy, double dt);
  bool getMileage();
  void processBattery();
  void processLoop();
  void joyLoop();
  void writeLoop();
  void setWriteCS(int len);
  bool checkCS(int len);
  bool checkSN();
  void processOdometry();
  void processIMU(uint8_t head_type);
  void checkSN(int type);  // for imu
  void imuCallback(const sensor_msgs::Imu::ConstPtr& msg);
  void quaternionToEuler(double Qw, double Qx, double Qy, double Qz, double &pitch, double &roll, double &yaw); 
  float invSqrt(float number);
  void MahonyAHRSupdateIMU(float q[4], float gx, float gy, float gz, float ax, float ay, float az, float delta_s);
  void magCalculateYaw(double roll, double pitch, double &magyaw, double magx, double magy, double magz);
  void setSerial();
  void openSerial();
  void callHandle();
  void updateSN();
  bool read_msg();
  ros::NodeHandle nh_;

private:
  bool write_msg(double linear_x, double linear_y, double angular_z);

  boost::thread* pJoyThread_;
  boost::thread* processThread_;
  boost::thread* writeThread_;
  boost::recursive_mutex Control_mutex_;
  
  //version
  std::string ws_version_;
  std::string hw_version_;
  std::string base_type_name_;

  //setting
  bool provide_odom_tf_, debug_log_;
  int controll_type_ ;
  bool joy_enable_;

  int encode_resolution_;
  double wheel_radius_;
  double period_;
  double base_shape_a_,base_shape_b_;

  //position
  
  //moving info
  double  linear_speed_min_;
  double angular_speed_min_;
  double  linear_speed_max_;
  double angular_speed_max_;
  
  //sum info
  double Mileage_sum_;
    double Mileage_last_;
  int sn_lost_  = 0;
  int cs_error_ = 0;
  uint32_t write_sn_ = 0;
  uint32_t read_sn_  = 0;
  
  //flages 
  bool read_first_;
  bool imu_frist_sn_;

  //serial
  std::string port_;
  int baud_;
  int serial_timeout_;
  int rate_;
  double duration_;

  //pose
  double x_,y_,th_;
  //vel
  nav_msgs::Odometry current_odom_;

  // PID 闭环控制
  double kp_vx_, ki_vx_, kd_vx_;
  double kp_vy_, ki_vy_, kd_vy_;
  double kp_vth_, ki_vth_, kd_vth_;
  double integral_vx_, integral_vy_, integral_vth_;
  double last_error_vx_, last_error_vy_, last_error_vth_;
  double actual_vx_, actual_vy_, actual_vth_;
  bool enable_pid_;

  // Acceleration ramping (slew rate limiting)
  double max_accel_x_, max_accel_y_, max_accel_th_;
  double prev_ramp_linear_x_, prev_ramp_linear_y_, prev_ramp_angular_z_;
  // Low-pass filter on cmd_vel
  double cmd_vel_filter_alpha_;
  double filtered_cmd_linear_x_, filtered_cmd_linear_y_, filtered_cmd_angular_z_;
  // Configurable integral anti-windup limits
  double integral_limit_vx_, integral_limit_vy_, integral_limit_vth_;
  // PID measured elapsed time
  ros::Time last_pid_time_;

  //battery
  float current_battery_percent_;
  //led_values
  int   led_mode_type_;
  float led_frequency_;
  float led_red_value_;
  float led_green_value_;
  float led_blue_value_;
  double led_t_0;
  int led_timer;

  //data
  pack_write pack_write_;
  pack_read  pack_read_;

    //fdlink data
  FDILink::imu_frame_read  imu_frame_;
  FDILink::ahrs_frame_read ahrs_frame_;
  FDILink::insgps_frame_read insgps_frame_;

  //joy ctl
  double linear_gain_;
  double twist_gain_;
  double joy_linear_x_,  joy_linear_y_,  joy_angular_z_;
  double cmd_linear_x_,  cmd_linear_y_,  cmd_angular_z_;
  double move_linear_x_, move_linear_y_, move_angular_z_;
  double cmd_dt_threshold_;

  string Mileage_file_name_;
  string Mileage_backup_file_name_;

  
  //frame name
  string base_frame_, odom_frame_;
  string imu_frame_id_;

  //topic
  string vel_topic_, joy_topic_, odom_topic_;
  string battery_topic_;
  string imu_topic_, mag_pose_2d_topic_;

  //rostimer
  ros::Time current_time_, last_time_;
  ros::Time last_cmd_time_;

  //Publisher
  ros::Publisher odom_pub_;
  ros::Publisher mileage_pub_;
  ros::Publisher battery_pub_;
  ros::Publisher imu_pub_;
  ros::Publisher mag_pose_pub_;
  //Subscriber
  ros::Subscriber vel_sub_, joy_sub_;

  ros::ServiceServer set_max_vel_server_;
  ros::ServiceServer get_max_vel_server_;

  ros::ServiceServer stop_move_server_;
  ros::ServiceServer get_battery_state_server_;
  ros::ServiceServer set_led_server_;

  tf::TransformBroadcaster odom_broadcaster_;

  serial::Serial serial_; //声明串口对象

};//baseBringup
} //ucarController


#endif
