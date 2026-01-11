#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2


class StopLineDetector:
    """
    gray_label(라벨 이미지)을 받아 stopline 여부(True/False) 판단.

    - mode 0,1: 노란 정지선(라벨=128)
    - mode 2  : 흰 정지선(라벨=255)

    ROI:
      y: [0.60H, 0.80H)
      x: [0.20W, 0.80W)

    반환:
      detected(bool), viz_img(or None), debug(dict)
    """

    YELLOW_LABEL = 128
    WHITE_LABEL = 255

    def __init__(
        self,
        roi_y1_ratio: float = 0.60,
        roi_y2_ratio: float = 0.80,
        roi_x1_ratio: float = 0.20,
        roi_x2_ratio: float = 0.80,
        # 감지 임계값(ROI 내 비율)
        yellow_ratio_thresh: float = 0.05,
        white_ratio_thresh: float = 0.05,
    ):
        self.ry1 = float(roi_y1_ratio)
        self.ry2 = float(roi_y2_ratio)
        self.rx1 = float(roi_x1_ratio)
        self.rx2 = float(roi_x2_ratio)

        self.yellow_ratio_thresh = float(yellow_ratio_thresh)
        self.white_ratio_thresh = float(white_ratio_thresh)

    def _get_roi(self, gray_label: np.ndarray):
        h, w = gray_label.shape[:2]
        y1 = int(h * self.ry1)
        y2 = int(h * self.ry2)
        x1 = int(w * self.rx1)
        x2 = int(w * self.rx2)

        # 안전 클램프
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))

        # y2<=y1 같은 경우 방지
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        if x2 <= x1:
            x2 = min(w, x1 + 1)

        roi = gray_label[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)

    def detect(self, gray_label: np.ndarray, mode: int, enable_viz: bool = True):
        """
        main에서 호출:
          detected, viz, debug = stopline_det.detect(gray_label, mode, enable_viz=True)

        gray_label: 1채널 라벨 이미지 (dtype은 uint8 권장)
        mode: main의 Modes 값 (0/1/2에 반응)
        """
        debug = {
            "ok": False,
            "target": None,          # "yellow" or "white"
            "ratio": 0.0,
            "thresh": 0.0,
            "roi": None,
        }

        if gray_label is None:
            debug["ok"] = False
            return False, None, debug

        if gray_label.ndim != 2:
            # 혹시 3채널로 들어오면 1채널로 변환
            gray_label = cv2.cvtColor(gray_label, cv2.COLOR_BGR2GRAY)

        roi, (x1, y1, x2, y2) = self._get_roi(gray_label)
        debug["roi"] = (x1, y1, x2, y2)

        # 타겟 라벨 / 임계값 선택
        if mode in (0, 1):
            target_val = self.YELLOW_LABEL
            ratio_thresh = self.yellow_ratio_thresh
            debug["target"] = "yellow"
        elif mode == 2:
            target_val = self.WHITE_LABEL
            ratio_thresh = self.white_ratio_thresh
            debug["target"] = "white"
        else:
            # 다른 모드에서는 stopline 판단 안 함
            return False, None, debug

        # ROI 내 타겟 픽셀 비율 계산
        total = roi.size
        if total <= 0:
            return False, None, debug

        target_cnt = int(np.count_nonzero(roi == target_val))
        ratio = float(target_cnt) / float(total)

        detected = ratio >= ratio_thresh

        debug["ok"] = True
        debug["ratio"] = ratio
        debug["thresh"] = ratio_thresh

        # 시각화: 라벨 이미지를 컬러로 만들어 ROI 박스 표시
        viz = None
        if enable_viz:
            viz = cv2.cvtColor(gray_label, cv2.COLOR_GRAY2BGR)

            color = (0, 255, 0) if detected else (0, 0, 255)  # True면 초록, 아니면 빨강
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)

            txt = f"{debug['target']} ratio={ratio:.3f} thr={ratio_thresh:.3f} -> {detected}"
            cv2.putText(
                viz, txt, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
            )

        return detected, viz, debug
