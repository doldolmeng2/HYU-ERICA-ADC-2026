#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError


class SensorsTestNode:
    def __init__(self):
        rospy.init_node("sensors_test", anonymous=False)

        # Params (필요하면 rosparam으로 조절 가능)
        self.image_topic = rospy.get_param("~image_topic", "/camera/image_raw")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")

        # 라이다 시각화 설정
        self.max_range = float(rospy.get_param("~max_range", 1.5))  # m
        self.img_size = int(rospy.get_param("~lidar_img_size", 600))  # px
        self.center = self.img_size // 2
        self.m_per_px = self.max_range / (self.center - 10)  # 단순 스케일

        self.bridge = CvBridge()

        self.last_cam = None
        self.last_lidar = None

        rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1)
        rospy.Subscriber(self.scan_topic, LaserScan, self.cb_scan, queue_size=1)

        rospy.loginfo("sensors_test node started.")
        rospy.loginfo("Subscribing camera: %s", self.image_topic)
        rospy.loginfo("Subscribing lidar:  %s", self.scan_topic)

        self.loop()

    def cb_image(self, msg):
        try:
            # 대부분 bgr8로 변환 가능 (안되면 rgb8로 바꿔도 됨)
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.last_cam = cv_img
        except CvBridgeError as e:
            rospy.logwarn("CvBridge error: %s", str(e))

    def cb_scan(self, msg):
        # LaserScan -> (x,y) 점들로 변환해서 이미지에 그리기
        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(len(ranges), dtype=np.float32) * msg.angle_increment

        # 유효한 range만 사용
        valid = np.isfinite(ranges)
        valid &= (ranges > max(msg.range_min, 0.05))
        valid &= (ranges < min(msg.range_max, self.max_range))

        r = ranges[valid]
        a = angles[valid]

        x = r * np.cos(a)
        y = r * np.sin(a)

        # 시각화용 캔버스
        canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # 중심점(차량 위치)
        cv2.circle(canvas, (self.center, self.center), 4, (255, 255, 255), -1)

        # 축 표시(앞쪽 x+, 오른쪽 y- 처럼 보이게 y축 뒤집어서 그릴거임)
        # x축(앞)
        cv2.line(canvas, (self.center, self.center), (self.center + 60, self.center), (80, 80, 80), 1)
        # y축(왼쪽)
        cv2.line(canvas, (self.center, self.center), (self.center, self.center - 60), (80, 80, 80), 1)

        # 점들을 픽셀로 변환
        px = (x / self.m_per_px).astype(np.int32) + self.center
        py = self.center - (y / self.m_per_px).astype(np.int32)  # y 뒤집기

        # 화면 범위 안에 있는 점만
        inside = (px >= 0) & (px < self.img_size) & (py >= 0) & (py < self.img_size)
        px = px[inside]
        py = py[inside]

        # 점 찍기
        for i in range(len(px)):
            cv2.circle(
                canvas,
                (px[i], py[i]),   # 중심 좌표
                2,                # radius (이 값이 점 크기)
                (0, 255, 0),      # 색
                -1                # 채우기 (-1 = filled)
            )


        step = float(rospy.get_param("~ring_step", 0.5))
        meter = step
        while meter <= self.max_range + 1e-6:
            rad = int(meter / self.m_per_px)
            cv2.circle(canvas, (self.center, self.center), rad, (40, 40, 40), 1)
            meter += step
        canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
        self.last_lidar = canvas

    def loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.last_cam is not None:
                cv2.imshow("Camera (/camera/image_raw)", self.last_cam)

            if self.last_lidar is not None:
                cv2.imshow("LiDAR (/scan) top-view", self.last_lidar)

            # OpenCV 창 갱신 (필수)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rospy.loginfo("Pressed q. Shutting down.")
                rospy.signal_shutdown("User quit")

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        SensorsTestNode()
    except rospy.ROSInterruptException:
        pass
