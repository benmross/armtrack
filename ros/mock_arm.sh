#!/usr/bin/env bash
# Start the team's arm URDF on mock hardware with only the stock controllers,
# so the bridge can be tested on any machine with ROS 2 Jazzy and ros2_control.
# Usage: mock_arm.sh <path to software-urc4 checkout>
set -euo pipefail
URC4=${1:?path to software-urc4}
WS=/tmp/mock_ws
mkdir -p $WS/src && ln -sfn "$URC4/src/description" $WS/src/description
set +u; source /opt/ros/jazzy/setup.bash
(cd $WS && colcon build --packages-select description >/dev/null)
source $WS/install/setup.bash
xacro "$(ros2 pkg prefix description)/share/description/urdf/arm/urc4_arm.urdf.xacro" \
  use_mock_hardware:=true deactivate_talon:=true sim_gazebo_classic:=false > /tmp/arm.urdf
cat > /tmp/mock_controllers.yaml <<YAML
controller_manager:
  ros__parameters:
    update_rate: 100
    joint_state_broadcaster: {type: joint_state_broadcaster/JointStateBroadcaster}
    joint_trajectory_controller: {type: joint_trajectory_controller/JointTrajectoryController}
joint_trajectory_controller:
  ros__parameters:
    joints: [base_yaw, shoulder_pitch, elbow_pitch, wrist_pitch, wrist_roll, gripper_claw]
    command_interfaces: [position]
    state_interfaces: [position, velocity]
    allow_partial_joints_goal: true
    open_loop_control: true
YAML
# Jazzy's controller manager takes the URDF from robot_state_publisher's topic
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$(cat /tmp/arm.urdf)" &
ros2 run controller_manager ros2_control_node --ros-args --params-file /tmp/mock_controllers.yaml &
sleep 3
ros2 run controller_manager spawner joint_state_broadcaster joint_trajectory_controller
wait
