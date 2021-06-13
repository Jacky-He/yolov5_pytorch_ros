import rospy 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os, cv2
from rospkg import RosPack

package = RosPack()
package_path = package.get_path('yolov5_pytorch_ros')

def publish():
    pub = rospy.Publisher('/camera/rgb_color', Image, queue_size=1)
    rospy.init_node('camera', anonymous=True)
    rate = rospy.Rate(1)
    bridge = CvBridge()
    while not rospy.is_shutdown():
        path = os.path.join(package_path, "scripts", "P0000.png")
        cv_image = cv2.imread(path)
        img_msg = bridge.cv2_to_imgmsg(cv_image, "bgr8")
        pub.publish(img_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish()
    except rospy.ROSInterruptException:
        pass
