#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, LaserScan
# [수정] Int16MultiArray 사용
from std_msgs.msg import Int16MultiArray 

class MainProcessor:
    def __init__(self):
        rospy.init_node('main_proc_node')
        self.bridge = CvBridge()
        
        self.drive_pub = rospy.Publisher('/motor', Int16MultiArray, queue_size=1)
        
        self.fix_speed = 98             
        self.steer_angles = [60, 90, 120] 
        self.steer_index = 0            
        self.last_change_time = rospy.get_time() 
        self.change_interval = 2.0      

        self.lidar_points = []
        self.W = 600
        self.H = 600
        self.CENTER = (self.W // 2, self.H // 2)
        self.SCALE = 180 

        self.image_sub = rospy.Subscriber('camera/image_raw', Image, self.image_callback)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        
        rospy.loginfo("메인 프로세서 시작 (Int16 사용)")

    def lidar_callback(self, msg):
        pts = []
        angle = msg.angle_min
        ROT = math.pi / 2 

        for r in msg.ranges:
            if msg.range_min < r < msg.range_max and (not math.isinf(r)):
                theta = angle
                theta2 = theta - ROT
                x = r * math.cos(theta2)
                y = r * math.sin(theta2)
                pts.append((x, y))
            angle += msg.angle_increment
        self.lidar_points = pts

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # --- 모터 제어 로직 ---
            now = rospy.get_time()
            if now - self.last_change_time >= self.change_interval:
                self.steer_index = (self.steer_index + 1) % len(self.steer_angles)
                self.last_change_time = now
            
            current_steer = self.steer_angles[self.steer_index]
            
            # [수정] Int16MultiArray 메시지 생성 및 발행
            drive_msg = Int16MultiArray()
            drive_msg.data = [current_steer, self.fix_speed]
            self.drive_pub.publish(drive_msg)
            
            # --- 시각화 ---
            cv2.putText(cv_image, f"Steer: {current_steer}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Camera View", cv_image)

            lidar_img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            cv2.line(lidar_img, (self.CENTER[0], 0), (self.CENTER[0], self.H), (60, 60, 60), 1)
            cv2.line(lidar_img, (0, self.CENTER[1]), (self.W, self.CENTER[1]), (60, 60, 60), 1)
            for r in [0.5, 1.0, 1.5, 2.0]:
                cv2.circle(lidar_img, self.CENTER, int(r * self.SCALE), (40, 40, 40), 1)

            for (x, y) in self.lidar_points:
                px = int(self.CENTER[0] + x * self.SCALE)
                py = int(self.CENTER[1] - y * self.SCALE)
                dist = math.sqrt(x*x + y*y)
                if dist < 0.3: color = (0, 0, 255)
                elif dist < 1.0: color = (0, 255, 255)
                else: color = (0, 255, 0)
                cv2.circle(lidar_img, (px, py), 2, color, -1)
            
            cv2.imshow("LiDAR View", lidar_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rospy.signal_shutdown("User Exit")

        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    try:
        processor = MainProcessor()
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
