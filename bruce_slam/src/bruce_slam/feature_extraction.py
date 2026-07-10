#!/usr/bin/env python3
import numpy as np
import cv2
from sensor_msgs.msg import PointCloud2, Image
import cv_bridge

from bruce_slam.utils.io import *
from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import *
from bruce_slam import pcl
from bruce_slam.sensors import SONAR_ADAPTERS, make_adapter
from scipy.interpolate import interp1d

from .utils import *
from .sonar import *

from bruce_slam.CFAR import CFAR


class FeatureExtraction(BruceNode):
    '''Class to handle extracting features from Sonar images using CFAR
    subsribes to the sonar driver and publishes a point cloud
    '''

    def __init__(self):
        '''Class constructor, no args required all read from yaml file
        '''

        #oculus info
        self.oculus = OculusProperty()

        #default parameters for CFAR
        self.Ntc = 40
        self.Ngc = 10
        self.Pfa = 1e-2
        self.rank = None
        self.alg = "SOCA"
        self.detector = None
        self.threshold = 0
        self.cimg = None

        #default parameters for point cloud
        self.colormap = "RdBu_r"
        self.pub_rect = True
        self.resolution = 0.5
        self.outlier_filter_radius = 1.0
        self.outlier_filter_min_points = 5
        self.skip = 5

        # for offline visualization
        self.feature_img = None

        #for remapping from polar to cartisian
        self.res = None
        self.height = None
        self.rows = None
        self.width = None
        self.cols = None
        self.map_x = None
        self.map_y = None
        self.f_bearings = None
        self.REVERSE_Z = 1
        self.maxRange = None

        # frame counter for skip logic on drivers without a ping id
        self.frame_count = 0

        #which vehicle is being used
        self.compressed_images = True

        # place holder for the multi-robot system
        self.rov_id = ""

    def configure(self):
        '''Calls the CFAR class constructor for the featureExtraction class
        '''
        self.detector = CFAR(self.Ntc, self.Ngc, self.Pfa, self.rank)

    def init_node(self, node_name="feature_extraction_node", **node_kwargs):

        # initialise the underlying rclpy node
        BruceNode.__init__(self, node_name, **node_kwargs)

        #read in CFAR parameters
        self.Ntc = self.get_param("CFAR/Ntc")
        self.Ngc = self.get_param("CFAR/Ngc")
        self.Pfa = self.get_param("CFAR/Pfa")
        self.rank = self.get_param("CFAR/rank")
        self.alg = self.get_param("CFAR/alg", "SOCA")
        self.threshold = self.get_param("filter/threshold")

        #read in PCL downsampling parameters
        self.resolution = self.get_param("filter/resolution")
        self.outlier_filter_radius = self.get_param("filter/radius")
        self.outlier_filter_min_points = self.get_param("filter/min_points")

        #parameter to decide how often to skip a frame
        self.skip = self.get_param("filter/skip")

        #are the incoming images compressed?
        self.compressed_images = self.get_param("compressed_images")

        #cv bridge
        self.BridgeInstance = cv_bridge.CvBridge()

        #read in the format
        self.coordinates = self.get_param(
            "visualization/coordinates", "cartesian"
        )

        #vis parameters
        self.radius = self.get_param("visualization/radius")
        self.color = self.get_param("visualization/color")

        # sonar driver (pluggable adapter + configurable topic). If sonar/driver
        # is unset it defaults to the Oculus adapter matching compressed_images;
        # the default TOPIC then follows the resolved driver (not compressed_images)
        # so the two knobs cannot silently disagree.
        sonar_driver = self.get_param("sonar/driver", "")
        if not sonar_driver:
            sonar_driver = "oculus_compressed" if self.compressed_images else "oculus_uncompressed"
        sonar_topic = self.get_param("sonar/topic", "")
        if not sonar_topic:
            if sonar_driver == "oculus_compressed":
                sonar_topic = SONAR_TOPIC
            elif sonar_driver == "oculus_uncompressed":
                sonar_topic = SONAR_TOPIC_UNCOMPRESSED
            else:
                raise ValueError(
                    "sonar/topic must be set explicitly for sonar driver '{}'".format(sonar_driver)
                )
        self.sonar_adapter, sonar_type = make_adapter(SONAR_ADAPTERS, sonar_driver, self)
        # exposed so the offline bag pump can route on the actual configured topic
        self.sonar_topic = sonar_topic

        #sonar subsciber — best-effort matches SensorDataQoS publishers
        # (sonar_proc's proc_sonar / the oculus driver) and still receives
        # reliable ones
        from rclpy.qos import qos_profile_sensor_data
        self.sonar_sub = self.create_subscription(
            sonar_type, sonar_topic, self.callback, qos_profile_sensor_data)

        #feature publish topic
        self.feature_pub = self.create_publisher(
            PointCloud2, SONAR_FEATURE_TOPIC, 10)

        #vis publish topic
        self.feature_img_pub = self.create_publisher(
            Image, SONAR_FEATURE_IMG_TOPIC, 10)

        self.configure()

    def generate_map_xy(self, ping):
        '''Generate a mesh grid map for the sonar image, this enables converison to cartisian from the
        source polar images

        ping: OculusPing message
        '''

        #get the parameters from the ping message (bearings are in radians)
        _res = ping.range_resolution
        _height = ping.num_ranges * _res
        _rows = ping.num_ranges
        _width = np.sin(
            (ping.bearings[-1] - ping.bearings[0]) / 2) * _height * 2
        _cols = int(np.ceil(_width / _res))

        #check if the parameters have changed
        if self.res == _res and self.height == _height and self.rows == _rows and self.width == _width and self.cols == _cols:
            return

        #if they have changed do some work
        self.res, self.height, self.rows, self.width, self.cols = _res, _height, _rows, _width, _cols

        #generate the mapping (bearings already in radians)
        bearings = np.asarray(ping.bearings, dtype=np.float32)
        f_bearings = interp1d(
            bearings,
            range(len(bearings)),
            kind='linear',
            bounds_error=False,
            fill_value=-1,
            assume_sorted=True)

        #build the meshgrid
        XX, YY = np.meshgrid(range(self.cols), range(self.rows))
        x = self.res * (self.rows - YY)
        y = self.res * (-self.cols / 2.0 + XX + 0.5)
        b = np.arctan2(y, x) * self.REVERSE_Z
        r = np.sqrt(np.square(x) + np.square(y))
        self.map_y = np.asarray(r / self.res, dtype=np.float32)
        self.map_x = np.asarray(f_bearings(b), dtype=np.float32)

    def publish_features(self, stamp, points):
        '''Publish the feature message using the provided parameters
        stamp: the time stamp of the source sonar image
        points: points to be converted to a ros point cloud, in cartisian meters
        '''

        #shift the axis
        points = np.c_[points[:,0],np.zeros(len(points)),  points[:,1]]

        #convert to a pointcloud
        feature_msg = n2r(points, "PointCloudXYZ")

        #give the feature message the same time stamp as the source sonar image
        #this is CRITICAL to good time sync downstream
        feature_msg.header.stamp = stamp
        feature_msg.header.frame_id = "base_link"

        #publish the point cloud, to be used by SLAM
        self.feature_pub.publish(feature_msg)

    #@add_lock
    def callback(self, sonar_msg):
        '''Feature extraction callback
        sonar_msg: the raw sonar driver message; normalized to a SonarPing (polar,
        grayscale image + geometry) via the configured sonar adapter.
        '''

        # cheap skip test BEFORE the adapter runs, so skipped frames never pay
        # the image decode. Drivers without a ping id use a message counter.
        self.frame_count += 1
        ping_id = getattr(sonar_msg, "ping_id", self.frame_count)
        if ping_id % self.skip != 0:
            self.feature_img = None
            # Don't extract features in every frame.
            # But we still need empty point cloud for synchronization in SLAM node.
            nan = np.array([[np.nan, np.nan]])
            self.publish_features(sonar_msg.header.stamp, nan)
            return

        #normalize the driver message to a SonarPing
        ping = self.sonar_adapter(sonar_msg)

        #the adapter already decoded the polar image to grayscale
        img = ping.image

        #generate a mesh grid mapping from polar to cartisian
        self.generate_map_xy(ping)

        # Detect targets and check against threshold using CFAR (in polar coordinates)
        peaks = self.detector.detect(img, self.alg)
        peaks &= img > self.threshold

        vis_img = cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)
        vis_img = cv2.applyColorMap(vis_img, 2)
        self.feature_img_pub.publish(self.BridgeInstance.cv2_to_imgmsg(vis_img, encoding="bgr8"))

        #convert to cartisian
        peaks = cv2.remap(peaks, self.map_x, self.map_y, cv2.INTER_LINEAR)
        locs = np.c_[np.nonzero(peaks)]

        #convert from image coords to meters
        x = locs[:,1] - self.cols / 2.
        x = (-1 * ((x / float(self.cols / 2.)) * (self.width / 2.))) #+ self.width
        y = (-1*(locs[:,0] / float(self.rows)) * self.height) + self.height
        points = np.column_stack((y,x))

        #filter the cloud using PCL
        if len(points) and self.resolution > 0:
            points = pcl.downsample(points, self.resolution)

        #remove some outliars
        if self.outlier_filter_min_points > 1 and len(points) > 0:
            # points = pcl.density_filter(points, 5, self.min_density, 1000)
            points = pcl.remove_outlier(
                points, self.outlier_filter_radius, self.outlier_filter_min_points
            )

        #publish the feature message
        self.publish_features(ping.header.stamp, points)
