#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2


class LineOffsetEstimator:
    """
    ROI 기반 blob 검출 + offset 계산 + 시각화

    입력:
      - gray_label: 0/128/255 (mask.py의 gray_label)
      - mode: 모드
      - frame_bgr: 원본 프레임(BGR) (시각화용, 없으면 label로 대체)
      - enable_viz: 시각화 이미지 반환 여부

    출력:
      - offset(float)
      - debug(dict)
      - viz_bgr(or None)

    mode 0:
      - ROI(띠 영역)에서 yellow(128) 연결요소(blob)들을 찾음
      - 각 blob의 중심(cx)을 계산
      - 가장 오른쪽 blob 중심을 선택
      - offset = target_x_right - chosen_cx
    """

    def __init__(self):
        # ====== ROI 파라미터 ======
        # ROI를 "비율"로 잡아두면 해상도 바뀌어도 자동으로 대응됨
        # 예: 하단 40%~80% 구간에서 차선이 잘 보인다고 가정
        self.roi_y_start_ratio = 0.55
        self.roi_y_end_ratio   = 0.85

        # ====== 대상 레이블 값 ======
        self.yellow_value = 128

        # ====== 목표 x ======
        # 640폭 기준 580이었으니 비율로도 가능: 580/640 ≈ 0.90625
        self.target_x_ratio = 0.90625  # 기본: w*0.90625
        self.target_x_right = None     # None이면 ratio 사용, 숫자 지정하면 그 값 사용

        # ====== 노이즈 제거 ======
        self.min_area = 80     # ROI 내 blob 최소 면적(픽셀 수)
        self.morph_open = 3    # 모폴로지 open 커널 크기(0이면 안함)

        # ====== 시각화 ======
        self.draw_radius = 6

    def get_offset(self, gray_label: np.ndarray, mode: int, frame_bgr=None, enable_viz: bool = False):
        debug = {"mode": mode}
        if gray_label is None:
            debug["reason"] = "gray_label is None"
            return 0.0, debug, None

        if mode == 0:
            return self._mode0_roi_right_yellow(gray_label, debug, frame_bgr, enable_viz)
        elif mode == 1:
            debug["reason"] = "mode1 not implemented yet"
            return 0.0, debug, self._make_empty_viz(frame_bgr, gray_label, enable_viz)
        elif mode == 2:
            debug["reason"] = "mode2 not implemented yet"
            return 0.0, debug, self._make_empty_viz(frame_bgr, gray_label, enable_viz)
        else:
            debug["reason"] = "unknown mode"
            return 0.0, debug, self._make_empty_viz(frame_bgr, gray_label, enable_viz)

    def _make_empty_viz(self, frame_bgr, gray_label, enable_viz: bool):
        if not enable_viz:
            return None
        if frame_bgr is not None:
            return frame_bgr.copy()
        return cv2.cvtColor(gray_label, cv2.COLOR_GRAY2BGR)

    def _mode0_roi_right_yellow(self, gray_label: np.ndarray, debug: dict, frame_bgr=None, enable_viz: bool = False):
        h, w = gray_label.shape[:2]

        # ROI 좌표 계산
        y0 = int(h * self.roi_y_start_ratio)
        y1 = int(h * self.roi_y_end_ratio)
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h, y1))
        if y1 <= y0 + 1:
            y0 = max(0, h - 120)
            y1 = h

        debug["img_w"] = w
        debug["img_h"] = h
        debug["roi_y0"] = y0
        debug["roi_y1"] = y1

        # 목표 x 설정
        if self.target_x_right is not None:
            target_x = int(np.clip(self.target_x_right, 0, w - 1))
        else:
            target_x = int(np.clip(w * self.target_x_ratio, 0, w - 1))

        debug["target_x"] = target_x

        # 시각화 바탕 준비
        viz = None
        if enable_viz:
            if frame_bgr is not None:
                viz = frame_bgr.copy()
            else:
                viz = cv2.cvtColor(gray_label, cv2.COLOR_GRAY2BGR)

            # ROI 박스 표시(초록)
            cv2.rectangle(viz, (0, y0), (w - 1, y1 - 1), (0, 255, 0), 2)

        # ROI에서 yellow 마스크 생성 (0/255)
        roi = gray_label[y0:y1, :]
        yellow_bin = (roi == self.yellow_value).astype(np.uint8) * 255

        # 모폴로지로 작은 점 노이즈 제거(선택)
        if self.morph_open and self.morph_open > 0:
            k = self.morph_open
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            yellow_bin = cv2.morphologyEx(yellow_bin, cv2.MORPH_OPEN, kernel)

        # 연결요소(connected components)로 blob 찾기
        # stats: [label, x, y, w, h, area]
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(yellow_bin, connectivity=8)

        blobs = []
        centers = []

        # label 0은 배경이니까 1부터
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if area < self.min_area:
                continue

            cx, cy = centroids[i]   # ROI 좌표계 기준
            cx_full = float(cx)            # x는 전체 폭 그대로
            cy_full = float(cy + y0)       # y는 ROI offset 반영

            blobs.append({
                "bbox": (int(x), int(y + y0), int(bw), int(bh)),
                "area": int(area),
                "cx": cx_full,
                "cy": cy_full,
            })
            centers.append(cx_full)

            # ✅ 시각화: 각 blob 중심 빨간 점
            if viz is not None:
                cv2.circle(viz, (int(cx_full), int(cy_full)), self.draw_radius, (0, 0, 255), -1)
                cv2.rectangle(viz, (int(x), int(y + y0)), (int(x + bw), int(y + y0 + bh)), (0, 0, 255), 1)

        debug["blob_count"] = len(blobs)
        debug["blobs"] = blobs

        # blob이 없으면 offset=0
        if len(centers) == 0:
            debug["reason"] = "no blobs after filtering"
            # 목표점은 찍어주기
            if viz is not None:
                y_ref = int((y0 + y1) * 0.5)
                cv2.circle(viz, (target_x, y_ref), self.draw_radius, (255, 0, 0), -1)
                cv2.putText(viz, "target", (target_x + 8, y_ref - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            return 0.0, debug, viz

        # 가장 오른쪽 blob 선택
        right_cx = float(max(centers))
        debug["chosen_right_cx"] = right_cx

        # offset = 목표 - 현재
        offset = float(target_x - right_cx)
        debug["offset"] = offset

        # ✅ 목표점(파란 점)은 ROI의 "중간 y"에 표시
        if viz is not None:
            y_ref = int((y0 + y1) * 0.5)
            # 목표점(파란)
            cv2.circle(viz, (target_x, y_ref), self.draw_radius, (255, 0, 0), -1)
            cv2.putText(viz, "target", (target_x + 8, y_ref - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

            # 선택된 blob 강조(노란 테두리)
            # (선택된 cx와 가장 가까운 blob의 cy를 찾는 게 더 예쁘지만, 여기선 단순화)
            # chosen 점은 y_ref에 찍어도 되고, 실제 blob의 중심에 찍어도 됨.
            # 여기선 "실제 blob 중심" 중 right_cx에 해당하는 blob을 찾아서 찍음.
            chosen_cy = y_ref
            for b in blobs:
                if abs(b["cx"] - right_cx) < 1e-3:
                    chosen_cy = int(b["cy"])
                    break

            cv2.circle(viz, (int(right_cx), int(chosen_cy)), self.draw_radius + 3, (0, 255, 255), 2)

            # 텍스트
            cv2.putText(viz, f"right_cx={right_cx:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(viz, f"target_x={target_x}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(viz, f"offset={offset:.1f}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        return offset, debug, viz
