# Reconstructed ROS 2 stub of the Rowe SeaPilot DVL message.
# Fields mirror the subset consumed by bruce_slam (velocity + altitude).
std_msgs/Header header
geometry_msgs/Vector3 velocity   # body-frame velocity (m/s)
float32 altitude                 # height above bottom (m)
