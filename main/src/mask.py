#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class MaskProcessor:
    """
    mask.py
    - main.py에서 import 해서 사용 (ROS 노드 아님)
    - 입력: BGR 이미지(np.ndarray)
    - 출력:
        yellow_mask: 0/255 (uint8)
        white_mask : 0/255 (uint8)  # 노란색 영역은 제거됨(우선순위)
        gray_label : 0/128/255 (uint8)  # 검정=0, 노랑=128, 흰=255
    """

    def __init__(self):
        # =========================
        # 기본 파라미터(너가 실험한 값)
        # 필요하면 main에서 set_params로 바꿀 수 있게 열어둠
        # =========================

        # Gaussian blur
        self.blur_ksize = (5, 5)
        self.blur_sigma = 0

        # Yellow (HLS) - 너가 준 기본값
        self.Y_H_min = 26
        self.Y_H_max = 33
        self.Y_L_min = 40
        self.Y_L_max = 220
        self.Y_S_min = 37
        self.Y_S_max = 255

        # White (HLS) - 너가 준 기본값
        self.W_H_min = 0
        self.W_H_max = 179
        self.W_L_min = 188
        self.W_L_max = 255
        self.W_S_min = 0
        self.W_S_max = 255

    def set_params(self, **kwargs):
        """
        (선택) 런타임에 파라미터 바꾸고 싶으면 사용
        예: mask_proc.set_params(Y_H_min=25, W_L_min=180)
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def process(self, frame_bgr: np.ndarray):
        """
        핵심 함수
        - frame_bgr: OpenCV BGR 이미지
        - return: yellow_mask, white_mask, gray_label
        """

        if frame_bgr is None:
            return None, None, None

        # 1) 블러(노이즈 완화)
        blur = cv2.GaussianBlur(frame_bgr, self.blur_ksize, self.blur_sigma)

        # 2) BGR -> HLS
        hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)

        # 3) Yellow mask (먼저)
        y_min = np.array([self.Y_H_min, self.Y_L_min, self.Y_S_min], dtype=np.uint8)
        y_max = np.array([self.Y_H_max, self.Y_L_max, self.Y_S_max], dtype=np.uint8)
        yellow_mask = cv2.inRange(hls, y_min, y_max)  # 0/255

        # 4) White mask (그 다음)
        w_min = np.array([self.W_H_min, self.W_L_min, self.W_S_min], dtype=np.uint8)
        w_max = np.array([self.W_H_max, self.W_L_max, self.W_S_max], dtype=np.uint8)
        white_mask = cv2.inRange(hls, w_min, w_max)  # 0/255

        # ✅ 노란색 우선: 노란색으로 잡힌 픽셀은 흰색에서 제거
        white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(yellow_mask))

        # 5) (추가 요구사항) gray_label 생성: 검정=0, 노랑=128, 흰=255
        #    우선순위는 이미 yellow > white로 만들어져 있어서 겹침 없음.
        gray_label = np.zeros_like(yellow_mask, dtype=np.uint8)
        gray_label[yellow_mask > 0] = 128
        gray_label[white_mask > 0] = 255
        
        return yellow_mask, white_mask, gray_label

    def white_pixels_many(
        self,
        gray_label: np.ndarray,
        thresh: int = 7000,
        roi_y_start_ratio: float = 0.55,
        roi_y_end_ratio: float = 1.0
    ) -> bool:
        """
        하단 ROI에서 흰색 픽셀(255)이 일정 개수 이상이면 True

        - gray_label: 0/128/255 grayscale label
        - thresh: 흰 픽셀 개수 임계값
        - roi_y_start_ratio: ROI 시작 비율 (기본 0.55)
        - roi_y_end_ratio: ROI 끝 비율 (기본 1.0)
        """

        if gray_label is None:
            return False

        h, w = gray_label.shape[:2]

        # ROI y 범위 계산
        y0 = int(h * roi_y_start_ratio)
        y1 = int(h * roi_y_end_ratio)

        y0 = max(0, min(h - 1, y0))
        y1 = max(y0 + 1, min(h, y1))

        roi = gray_label[y0:y1, :]

        # 흰색(255) 픽셀 수 세기
        white_count = int(np.sum(roi == 255))
        # print("white_count : ", white_count)
        # print("white_count: ", white_count)
        return white_count >= thresh
