#!/usr/bin/env bash
# Run slam benchmark over every rosbag2 directory in the current folder.
for i in `seq 1 10`; do
    for bagdir in ./*/; do
        timeout 5m ros2 launch bruce_slam slam.launch.py file:=`pwd`/$bagdir rviz:=false
        mv ./*.npz . 2>/dev/null || true
        sleep 3
    done
done
