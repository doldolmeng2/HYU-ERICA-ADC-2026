#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def start_node():
    rospy.init_node('img_publisher_node')

    pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)

    bridge = CvBridge()
    rate = rospy.Rate(30)

    # ✅ V4L2 백엔드로 여는 게 설정이 잘 먹힘
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        rospy.logerr("카메라를 열 수 없습니다.")
        return

    # ✅ 원하는 설정: MJPG + 640x480 + 30fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_CONTRAST, 145)
    cap.set(cv2.CAP_PROP_SATURATION, 200)


    # ✅ 실제로 적용됐는지 확인(중요)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((actual_fourcc >> 8*i) & 0xFF) for i in range(4)])

    rospy.loginfo(f"Camera actual: {actual_w}x{actual_h} @ {actual_fps:.1f}fps, FOURCC={fourcc_str}")

    rospy.loginfo("카메라 노드가 시작되었습니다. 전송 중...")

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            try:
                img_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                pub.publish(img_msg)
            except Exception as e:
                rospy.logerr(f"변환 오류: {e}")
        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass
