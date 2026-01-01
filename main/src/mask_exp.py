#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import time


class HLSMaskExp:
    def __init__(self):
        rospy.init_node("mask_exp")

        # 카메라 토픽(필요하면 실행 시 _image_topic:=... 로 변경 가능)
        self.image_topic = rospy.get_param("~image_topic", "/camera/image_raw")

        self.bridge = CvBridge()
        self.frame = None

        # 1) 트랙바 전용 창 (영상 출력 절대 안 함)
        self.ctrl_win = "HLS Control"
        cv2.namedWindow(self.ctrl_win, cv2.WINDOW_NORMAL)
        self._create_trackbars()

        # 2) 시각화 창들(각각 별도 창)
        cv2.namedWindow("raw", cv2.WINDOW_NORMAL)
        cv2.namedWindow("white_mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("yellow_mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("combined_mask", cv2.WINDOW_NORMAL)

        # (선택) 창이 겹치면 보기 불편하니까 위치 고정
        cv2.moveWindow(self.ctrl_win, 0, 0)
        cv2.moveWindow("raw", 450, 0)
        cv2.moveWindow("white_mask", 450, 350)
        cv2.moveWindow("yellow_mask", 900, 0)
        cv2.moveWindow("combined_mask", 900, 350)

        rospy.Subscriber(self.image_topic, Image, self._cb_img, queue_size=1)
        rospy.loginfo(f"[mask_exp] Subscribed: {self.image_topic}")

    def _create_trackbars(self):
        # Yellow (HLS)
        cv2.createTrackbar("Y_H_min", self.ctrl_win, 26, 179, lambda x: None)
        cv2.createTrackbar("Y_H_max", self.ctrl_win, 36, 179, lambda x: None)
        cv2.createTrackbar("Y_L_min", self.ctrl_win, 40, 255, lambda x: None)
        cv2.createTrackbar("Y_L_max", self.ctrl_win, 220, 255, lambda x: None)
        cv2.createTrackbar("Y_S_min", self.ctrl_win, 37, 255, lambda x: None)
        cv2.createTrackbar("Y_S_max", self.ctrl_win, 255, 255, lambda x: None)

        # White (HLS)
        cv2.createTrackbar("W_H_min", self.ctrl_win, 0, 179, lambda x: None)
        cv2.createTrackbar("W_H_max", self.ctrl_win, 179, 179, lambda x: None)
        cv2.createTrackbar("W_L_min", self.ctrl_win, 188, 255, lambda x: None)
        cv2.createTrackbar("W_L_max", self.ctrl_win, 255, 255, lambda x: None)
        cv2.createTrackbar("W_S_min", self.ctrl_win, 0, 255, lambda x: None)
        cv2.createTrackbar("W_S_max", self.ctrl_win, 255, 255, lambda x: None)

    def _cb_img(self, msg: Image):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"[mask_exp] cv_bridge error: {e}")

    def _g(self, name: str) -> int:
        return cv2.getTrackbarPos(name, self.ctrl_win)

    def spin(self):
        rate = rospy.Rate(30)


        while not rospy.is_shutdown():
            if self.frame is None:
                cv2.waitKey(1)
                rate.sleep()
                continue

            start_t = time.perf_counter() 

            frame = self.frame.copy()
            blur = cv2.GaussianBlur(frame, (5, 5), 0)
            # BGR -> HLS
            hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)

            # Yellow mask (먼저)
            y_min = np.array([self._g("Y_H_min"), self._g("Y_L_min"), self._g("Y_S_min")], dtype=np.uint8)
            y_max = np.array([self._g("Y_H_max"), self._g("Y_L_max"), self._g("Y_S_max")], dtype=np.uint8)
            yellow_mask = cv2.inRange(hls, y_min, y_max)

            # White mask (그 다음)
            w_min = np.array([self._g("W_H_min"), self._g("W_L_min"), self._g("W_S_min")], dtype=np.uint8)
            w_max = np.array([self._g("W_H_max"), self._g("W_L_max"), self._g("W_S_max")], dtype=np.uint8)
            white_mask = cv2.inRange(hls, w_min, w_max)

            # ✅ 노란색으로 검출된 픽셀은 흰색에서 제거 (yellow 우선)
            white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(yellow_mask))

            # Combined mask (이제 두 개가 겹치지 않음)
            combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

            # ✅ imshow 여러 개 = 창 여러 개 (트랙바 창과 완전 별개)
            cv2.imshow("raw", frame)
            cv2.imshow("white_mask", white_mask)
            cv2.imshow("yellow_mask", yellow_mask)
            # cv2.imshow("combined_mask", combined_mask)

            end_t = time.perf_counter()     # ⏱️ 끝
            proc_ms = (end_t - start_t) * 1000.0

            # print(f"[mask_exp] processing time: {proc_ms:.2f} ms")

            # q 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                rospy.signal_shutdown("user quit")
                break

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        HLSMaskExp().spin()
    except rospy.ROSInterruptException:
        pass