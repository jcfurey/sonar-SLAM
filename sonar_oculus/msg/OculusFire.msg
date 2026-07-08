# Reconstructed ROS 2 stub of the Blueprint Subsea Oculus "simple fire" message.
# Fields mirror the subset consumed by bruce_slam (see src/bruce_slam/sonar.py).
std_msgs/Header header
uint8 mode              # 1 = low frequency (navigation), 2 = high frequency (identification)
uint8 gamma             # gamma correction, 0/0xff => 1.0
uint8 flags             # bit flags, see Oculus SDK
float64 range           # range demand (meters or percent depending on flags)
float64 gain            # gain / intensity demand
float64 speed_of_sound  # m/s (0 => internal calc from salinity)
float64 salinity        # ppt
