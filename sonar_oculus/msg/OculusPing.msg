# Reconstructed ROS 2 stub of the Blueprint Subsea Oculus ping message
# (compressed image payload). Fields mirror the subset consumed by bruce_slam.
std_msgs/Header header
OculusFire fire_msg
int32 ping_id
uint32 part_number
uint32 start_time
uint16 num_ranges
uint16 num_beams
float32 range_resolution
int16[] bearings          # per-beam bearings, units of 0.01 degrees
sensor_msgs/Image ping    # sonar image (compressed payload in data)
