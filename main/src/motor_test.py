#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 2025/12/27 12:30 수정본

import rospy
import cv2
import sys
import signal
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Int16MultiArray

motor_pub = None

# =======================================================
# 센서 수신 클래스
# =======================================================
class SensorReceiver:
    def __init__(self):
        self.bridge = CvBridge()
        self.frame = None
        self.lidar_ranges = []

        rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback)

    def image_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(e)

    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges


# =======================================================
# 참가자 수정 구간 (제어 알고리즘)
# =======================================================
def calculate_control(frame, lidar_ranges):
    """
    frame         : OpenCV 이미지 (BGR)
    lidar_ranges  : 라이다 거리 배열
    return        : steer (45~135, 90 = 직진, 45 = 좌회전, 135 우회전), speed (90~119, 90~97 = 정지, 98+ = 직진)
    """

    steer = 135   # TODO: 조향 로직
    speed = 90   # TODO: 속도 로직

    return steer, speed
# =======================================================

def publish_neutral():
    global motor_pub
    if motor_pub is None:
        return

    neutral_msg = Int16MultiArray()
    neutral_msg.data = [90, 90]   # 중립
    motor_pub.publish(neutral_msg)
    rospy.loginfo("중립 신호 전송 (90, 90)")

def safe_exit(sig=None, frame=None):
    rospy.loginfo("종료 중... 중립 신호 전송")
    publish_neutral()
    rospy.sleep(0.1)   # 전송 보장용 짧은 대기
    cv2.destroyAllWindows()

# =======================================================
# 메인 노드
# =======================================================
def main():
    global motor_pub
    rospy.init_node("cam_lidar_motor_node")

    sensor = SensorReceiver()

    motor_pub = rospy.Publisher(
        "/motor",
        Int16MultiArray,
        queue_size=1
    )
    
    rospy.on_shutdown(safe_exit)
    
    rospy.sleep(0.5)      # 퍼블리셔 연결 안정화
    publish_neutral()     # ✅ s프로그램 시작 시 중립
    rospy.sleep(0.5)      # 중립 후 잠시 대기
    
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        if sensor.frame is None:
            rate.sleep()
            continue

        steer, speed = calculate_control(
            sensor.frame,
            sensor.lidar_ranges
        )

        motor_msg = Int16MultiArray()
        motor_msg.data = [steer, speed]
        motor_pub.publish(motor_msg)

        rate.sleep()


if __name__ == "__main__":
    main()
