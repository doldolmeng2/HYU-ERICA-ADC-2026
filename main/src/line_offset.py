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
      - offset = chosen_cx - target_x_right 
    """

    def __init__(self):
        # ====== ROI 파라미터 ======
        # ROI를 "비율"로 잡아두면 해상도 바뀌어도 자동으로 대응됨
        # 예: 하단 40%~80% 구간에서 차선이 잘 보인다고 가정
        self.roi_y_start_ratio = 0.65
        self.roi_y_end_ratio   = 0.70

        self.right_gate_ratio = 0.50
        self.fallback_offset = 150.0

        # ====== 대상 레이블 값 ======
        self.yellow_value = 128

        # ====== 목표 x ======
        # 640폭 기준 580이었으니 비율로도 가능: 580/640 ≈ 0.90625
        self.target_x_ratio = 0.78  # 기본: w*0.90625
        self.target_x_right = None     # None이면 ratio 사용, 숫자 지정하면 그 값 사용

        self.prev_offset = 0

        # ====== 노이즈 제거 ======
        self.min_area = 80     # ROI 내 blob 최소 면적(픽셀 수)
        self.morph_open = 3    # 모폴로지 open 커널 크기(0이면 안함)

        # ====== 시각화 ======
        self.draw_radius = 6

        # ====== 흰차선 너비값 ======
        self.lane_width_px = 460.0


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
            return self._mode2_roi_white_both_sides(gray_label, debug, frame_bgr, enable_viz)
        else:
            debug["reason"] = "unknown mode"
            return 0.0, debug, self._make_empty_viz(frame_bgr, gray_label, enable_viz)

    def _make_empty_viz(self, frame_bgr, gray_label, enable_viz: bool):
        if not enable_viz:
            return None
        if frame_bgr is not None:
            return frame_bgr.copy()
        return cv2.cvtColor(gray_label, cv2.COLOR_GRAY2BGR)

    def _mode0_roi_right_yellow(
        self,
        gray_label: np.ndarray,
        debug: dict,
        frame_bgr=None,
        enable_viz: bool = False
    ):
        """
        mode0: 오른쪽 노란 차선 기준 offset 계산 (정지선 강건 버전)

        - ROI에서 yellow(=self.yellow_value, 보통 128) blob 검출
        - 가로폭이 너무 큰 blob은 정지선으로 간주하고 무시
        - 남은 blob 중 가장 오른쪽(cx 최대) 선택
        - 단, 아래 경우에는 offset 업데이트 안 하고 이전 offset 유지
            1) 유효 blob 없음
            2) 가장 오른쪽 blob의 cx < w * right_gate_ratio (기본 0.6)
        """

        h, w = gray_label.shape[:2]

        # =============================
        # 1) ROI 설정
        # =============================
        y0 = int(h * self.roi_y_start_ratio)
        y1 = int(h * self.roi_y_end_ratio)
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h, y1))
        if y1 <= y0 + 1:
            y0 = max(0, h - 120)
            y1 = h

        debug["roi_y0"] = y0
        debug["roi_y1"] = y1

        # =============================
        # 2) 목표 x (target_x)
        # =============================
        if self.target_x_right is not None:
            target_x = int(np.clip(self.target_x_right, 0, w - 1))
        else:
            target_x = int(np.clip(w * self.target_x_ratio, 0, w - 1))

        debug["target_x"] = target_x

        # =============================
        # 3) 기준값들
        # =============================
        right_gate_ratio = getattr(self, "right_gate_ratio", 0.60)
        right_gate_x = w * right_gate_ratio

        max_blob_width_ratio = getattr(self, "max_blob_width_ratio", 0.65)
        max_blob_width = w * max_blob_width_ratio

        debug["right_gate_x"] = right_gate_x
        debug["max_blob_width"] = max_blob_width

        # =============================
        # 4) 시각화 캔버스
        # =============================
        viz = None
        if enable_viz:
            if frame_bgr is not None:
                viz = frame_bgr.copy()
            else:
                viz = cv2.cvtColor(gray_label, cv2.COLOR_GRAY2BGR)

            # ROI
            cv2.rectangle(viz, (0, y0), (w - 1, y1 - 1), (0, 255, 0), 2)
            # gate line
            gx = int(right_gate_x)
            cv2.line(viz, (gx, 0), (gx, h - 1), (0, 0, 255), 2)

        # =============================
        # 5) ROI에서 yellow mask 생성
        # =============================
        roi = gray_label[y0:y1, :]
        yellow_bin = (roi == self.yellow_value).astype(np.uint8) * 255

        if self.morph_open and self.morph_open > 0:
            k = self.morph_open
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            yellow_bin = cv2.morphologyEx(yellow_bin, cv2.MORPH_OPEN, kernel)

        # =============================
        # 6) blob 검출
        # =============================
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(
            yellow_bin, connectivity=8
        )

        centers = []
        blobs = []

        for i in range(1, num):
            x, y, bw, bh, area = stats[i]

            # 면적 필터
            if area < self.min_area:
                continue

            # ❗ 가로폭 필터 (정지선 제거)
            if bw > max_blob_width:
                debug.setdefault("ignored_blobs", []).append(
                    {"bw": int(bw), "area": int(area), "reason": "too wide"}
                )
                if viz is not None:
                    cv2.rectangle(
                        viz,
                        (int(x), int(y + y0)),
                        (int(x + bw), int(y + y0 + bh)),
                        (255, 0, 255),
                        3,
                    )
                continue

            cx, cy = centroids[i]
            cx_full = float(cx)
            cy_full = float(cy + y0)

            centers.append(cx_full)
            blobs.append({"cx": cx_full, "cy": cy_full, "bw": bw})

            if viz is not None:
                cv2.circle(viz, (int(cx_full), int(cy_full)), 5, (0, 0, 255), -1)

        debug["valid_blob_count"] = len(centers)

        # =============================
        # 7) fallback: 유효 blob 없음
        # =============================
        if len(centers) == 0:
            debug["fallback"] = True
            debug["reason"] = "no valid lane blob"
            debug["offset"] = self.prev_offset
            return self.prev_offset, debug, viz

        # =============================
        # 8) 가장 오른쪽 blob 선택
        # =============================
        right_cx = max(centers)
        debug["right_cx"] = right_cx

        # gate 기준 실패 → 이전 offset 유지
        if right_cx < right_gate_x:
            debug["fallback"] = True
            debug["reason"] = "rightmost blob too left"
            debug["offset"] = self.prev_offset
            return self.prev_offset, debug, viz

        # =============================
        # 9) 정상 offset 계산
        # =============================
        offset = float(right_cx - target_x)
        self.prev_offset = offset

        debug["fallback"] = False
        debug["offset"] = offset

        if viz is not None:
            y_ref = int((y0 + y1) * 0.5)
            cv2.circle(viz, (int(target_x), y_ref), 6, (255, 0, 0), -1)
            cv2.circle(viz, (int(right_cx), y_ref), 8, (0, 255, 255), 2)

            cv2.putText(
                viz,
                f"offset={offset:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        return offset, debug, viz

    def _mode2_roi_white_both_sides(
        self,
        gray_label: np.ndarray,
        debug: dict,
        frame_bgr=None,
        enable_viz: bool = False
    ):
        """
        mode2: 흰 차선(255) 기반으로 좌/우 차선을 잡아서 offset 계산

        - ROI(roi_y_start_ratio~roi_y_end_ratio)에서 흰색 blob들을 찾음
        - 화면 중앙(mid_x)을 기준으로
            left 후보: cx < mid_x
            right 후보: cx > mid_x
        로 분리
        - left는 left 후보 중 "가장 중앙에 가까운 것" (= cx가 최대인 left)
        - right는 right 후보 중 "가장 중앙에 가까운 것" (= cx가 최소인 right)
        - 둘 다 있으면 lane_center = (left + right)/2
        - 한쪽만 있으면 lane_width(픽셀, 고정값)로 lane_center 추정
            right만: lane_center = right - lane_width/2
            left만 : lane_center = left  + lane_width/2
        - offset = lane_center - mid_x
        - 실패(둘 다 없음) 시 이전 offset 유지
        """

        h, w = gray_label.shape[:2]
        mid_x = w * 0.5

        # =============================
        # 1) ROI 설정 (요청값)
        # =============================
        # 요청: 0.65 ~ 0.70
        y0 = int(h * 0.65)
        y1 = int(h * 0.70)
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h, y1))
        if y1 <= y0 + 1:
            y0 = max(0, h - 120)
            y1 = h

        debug["roi_y0"] = y0
        debug["roi_y1"] = y1
        debug["mid_x"] = mid_x

        # =============================
        # 2) 차선 폭(픽셀) 추정값
        # =============================
        # 사용자가 나중에 직접 넣을 거라 했으니 파라미터로 빼둠
        lane_width = float(getattr(self, "lane_width_px", 460.0))
        debug["lane_width_px"] = lane_width

        # =============================
        # 3) 시각화 캔버스
        # =============================
        viz = None
        if enable_viz:
            viz = frame_bgr.copy() if frame_bgr is not None else cv2.cvtColor(gray_label, cv2.COLOR_GRAY2BGR)

            # ROI 박스(초록)
            cv2.rectangle(viz, (0, y0), (w - 1, y1 - 1), (0, 255, 0), 2)

            # 중앙선(파랑)
            cv2.line(viz, (int(mid_x), 0), (int(mid_x), h - 1), (255, 0, 0), 2)

        # =============================
        # 4) ROI에서 흰색 마스크 생성
        # =============================
        roi = gray_label[y0:y1, :]
        white_bin = (roi == self.white_value).astype(np.uint8) * 255  # 0/255

        # 모폴로지 open (선택)
        if getattr(self, "morph_open", 0) and self.morph_open > 0:
            k = self.morph_open
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            white_bin = cv2.morphologyEx(white_bin, cv2.MORPH_OPEN, kernel)

        # =============================
        # 5) blob 검출
        # =============================
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(white_bin, connectivity=8)

        left_candidates = []   # (cx, cy, bbox, area)
        right_candidates = []

        all_blobs = []
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if area < self.min_area:
                continue

            cx, cy = centroids[i]          # ROI 좌표
            cx_full = float(cx)            # 전체 x 그대로
            cy_full = float(cy + y0)       # 전체 y로 변환

            bbox_full = (int(x), int(y + y0), int(bw), int(bh))

            all_blobs.append({"cx": cx_full, "cy": cy_full, "area": int(area), "bbox": bbox_full})

            # 중앙 기준 분리
            if cx_full < mid_x:
                left_candidates.append((cx_full, cy_full, bbox_full, int(area)))
            elif cx_full > mid_x:
                right_candidates.append((cx_full, cy_full, bbox_full, int(area)))

            # 시각화: 모든 blob 중심 빨강 점 + bbox
            if viz is not None:
                cv2.circle(viz, (int(cx_full), int(cy_full)), self.draw_radius, (0, 0, 255), -1)
                cv2.rectangle(
                    viz,
                    (bbox_full[0], bbox_full[1]),
                    (bbox_full[0] + bbox_full[2], bbox_full[1] + bbox_full[3]),
                    (0, 0, 255),
                    1
                )

        debug["blob_count"] = len(all_blobs)
        debug["all_blobs"] = all_blobs
        debug["left_candidate_count"] = len(left_candidates)
        debug["right_candidate_count"] = len(right_candidates)

        # =============================
        # 6) left/right 선택
        # =============================
        left = None
        right = None

        # left: 중앙에 가장 가까운(= 가장 오른쪽 left) => cx 최대
        if len(left_candidates) > 0:
            left = max(left_candidates, key=lambda t: t[0])  # (cx, cy, bbox, area)

        # right: 중앙에 가장 가까운(= 가장 왼쪽 right) => cx 최소
        if len(right_candidates) > 0:
            right = min(right_candidates, key=lambda t: t[0])

        debug["chosen_left"] = None if left is None else {"cx": left[0], "cy": left[1], "bbox": left[2], "area": left[3]}
        debug["chosen_right"] = None if right is None else {"cx": right[0], "cy": right[1], "bbox": right[2], "area": right[3]}

        # =============================
        # 7) lane_center 계산 + offset
        # =============================
        if left is None and right is None:
            # 둘 다 없음: 이전 offset 유지
            debug["fallback"] = True
            debug["reason"] = "no left/right lane detected"
            debug["lane_center_x"] = None
            debug["offset"] = self.prev_offset
            return self.prev_offset, debug, viz

        # 둘 다 있으면 평균
        if left is not None and right is not None:
            lane_center_x = 0.5 * (left[0] + right[0])
            debug["center_estimation"] = "both"

        # right만 있으면: lane_center = right - lane_width/2
        elif right is not None:
            lane_center_x = right[0] - lane_width * 0.5
            debug["center_estimation"] = "right_only"

        # left만 있으면: lane_center = left + lane_width/2
        else:
            lane_center_x = left[0] + lane_width * 0.5
            debug["center_estimation"] = "left_only"

        offset = float(lane_center_x - mid_x)

        # 정상 업데이트
        self.prev_offset = offset

        debug["fallback"] = False
        debug["lane_center_x"] = float(lane_center_x)
        debug["offset"] = offset

        # =============================
        # 8) 시각화 추가 (mode0 스타일)
        # =============================
        if viz is not None:
            y_ref = int((y0 + y1) * 0.5)

            # 이미지 중앙(파랑 점)
            cv2.circle(viz, (int(mid_x), y_ref), self.draw_radius, (255, 0, 0), -1)
            cv2.putText(viz, "img_center", (int(mid_x) + 8, y_ref - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

            # 추정 lane_center(노랑 점)
            cv2.circle(viz, (int(lane_center_x), y_ref), self.draw_radius, (0, 255, 255), -1)
            cv2.putText(viz, "lane_center", (int(lane_center_x) + 8, y_ref - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # left 선택 강조(초록 원)
            if left is not None:
                cv2.circle(viz, (int(left[0]), int(left[1])), self.draw_radius + 4, (0, 255, 0), 2)
                cv2.putText(viz, f"L={left[0]:.1f}", (int(left[0]) + 8, int(left[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # right 선택 강조(주황 원)
            if right is not None:
                cv2.circle(viz, (int(right[0]), int(right[1])), self.draw_radius + 4, (0, 165, 255), 2)
                cv2.putText(viz, f"R={right[0]:.1f}", (int(right[0]) + 8, int(right[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

            # 텍스트
            cv2.putText(viz, f"est={debug['center_estimation']}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(viz, f"lane_center_x={lane_center_x:.1f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(viz, f"offset={offset:.1f}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        return offset, debug, viz